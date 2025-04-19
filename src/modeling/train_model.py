import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
from datetime import timedelta
from io import BytesIO
import warnings
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union, List, Dict, Any
import yfinance as yf
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from prefect import task, flow

# Suppress warnings
warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]


class BTCTimeSeriesModel(mlflow.pyfunc.PythonModel):
    def __init__(self, p=None, d=None, q=None, seasonal=False, stepwise=True,
                 max_p=5, max_q=5):
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = seasonal
        self.stepwise = stepwise
        self.max_p = max_p
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
        self.train_index = None
        self.train_data = None
    
    def find_best_params(self, train_data):
        """Use auto_arima to find the best parameters"""
        model = auto_arima(train_data, 
                           seasonal=self.seasonal,  
                           stepwise=self.stepwise,
                           suppress_warnings=True,
                           d=self.d, 
                           max_p=self.max_p,
                           max_q=self.max_q)  
        
        # Extract best parameters
        order = model.order
        self.p, self.d, self.q = order
        return model
        
    def fit(self, train_data):
        """Train the model on the provided data"""
        self.train_index = train_data.index  # Save for plotting later
        self.train_data = train_data  # Save the training data
        
        if self.p is None or self.q is None:
            # Find best parameters with auto_arima
            self.model = self.find_best_params(train_data)
            self.fitted_model = self.model
        else:
            # Use user-specified parameters
            self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
            self.fitted_model = self.model.fit()
        
        return self
    
    def predict(self, context: mlflow.pyfunc.PythonModelContext, 
                model_input: List[Union[int, pd.DataFrame, List[Union[int, pd.DataFrame]]]]) -> List[Dict[str, float]]:
        """
        Predict method that handles multiple input types
        Args:
            context: MLflow model context
            model_input: List containing integers for future periods, DataFrames with data, or lists of either
        Returns:
            List of predictions as dictionaries with 'predicted_mean' key
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fit. Call fit() before predict().")
    
        if model_input is None:
            raise ValueError("Input data cannot be None")
        
        # Handle the list wrapper for model_input
        if len(model_input) == 1:
            actual_input = model_input[0]
        else:
            # If multiple items in the list, process them all
            results = []
            for item in model_input:
                # Recursive call with a single item list
                forecast = self.predict(context, [item])
                results.extend(forecast)
            return results
    
        # Now process the actual input
        if isinstance(actual_input, int):
            # Handle numeric input - predict step by step for the specified number of periods
            n_periods = actual_input
            if n_periods <= 0:
                raise ValueError("Number of periods must be positive")
            
            # For large numbers of steps, we need to predict iteratively
            predictions = []
            
            # If we have a pmdarima model, use its built-in prediction
            if hasattr(self.fitted_model, 'predict_in_sample') and hasattr(self.fitted_model, 'predict'):
                # This is a pmdarima model
                forecast = self.fitted_model.predict(n_periods=n_periods)
                return [{"forecast": float(x)} for x in forecast]
            else:
                # For statsmodels ARIMA
                # We need to create predictions one step at a time for large forecasts
                history = self.train_data.copy().values.flatten()
                
                for t in range(n_periods):
                    # Make one-step forecast
                    if hasattr(self.fitted_model, 'get_forecast'):
                        next_pred = self.fitted_model.get_forecast(steps=1).predicted_mean[0]
                    else:
                        next_pred = self.fitted_model.forecast(steps=1)[0]
                        
                    predictions.append({"forecast": float(next_pred)})
                    
                    # Add prediction to history for multi-step forecasting
                    history = np.append(history, next_pred)
                    
                    # Refit the model if needed for next step (may be necessary for some models)
                    if t > 0 and t % 100 == 0:  # Refit every 100 steps to avoid numerical issues
                        self.model = ARIMA(history, order=(self.p, self.d, self.q))
                        self.fitted_model = self.model.fit()
                
                return predictions
    
        elif isinstance(actual_input, pd.DataFrame):
            # Handle DataFrame input - forecast for each date in the DataFrame
            n_periods = len(actual_input)
            
            # Use same approach as integer input
            if hasattr(self.fitted_model, 'predict_in_sample') and hasattr(self.fitted_model, 'predict'):
                # This is a pmdarima model
                forecast = self.fitted_model.predict(n_periods=n_periods)
                return [{"predicted_mean": float(x)} for x in forecast]
            else:
                # For statsmodels ARIMA
                history = self.train_data.copy().values.flatten()
                predictions = []
                
                for t in range(n_periods):
                    if hasattr(self.fitted_model, 'get_forecast'):
                        next_pred = self.fitted_model.get_forecast(steps=1).predicted_mean[0]
                    else:
                        next_pred = self.fitted_model.forecast(steps=1)[0]
                        
                    predictions.append({"forecast": float(next_pred)})
                    
                    # Add prediction to history for multi-step forecasting
                    history = np.append(history, next_pred)
                    
                    # Refit the model if needed
                    if t > 0 and t % 100 == 0:
                        self.model = ARIMA(history, order=(self.p, self.d, self.q))
                        self.fitted_model = self.model.fit()
                
                return predictions
    
        elif isinstance(actual_input, list):
            results = []
            for item in actual_input:
                if isinstance(item, pd.DataFrame):
                    forecast = self.predict(context, [item])
                    results.extend(forecast)
                elif isinstance(item, int):
                    forecast = self.predict(context, [item])
                    results.extend(forecast)
                else:
                    raise TypeError(f"Unsupported input type in list: {type(item)}")
            return results
    
        else:
            raise TypeError(f"Unsupported input type: {type(actual_input)}")


    def generate_confidence_intervals(self, forecast, std_error_factor=1.96):
        """Generate 95% confidence intervals for the forecast"""
        # Check if aicc is a method or float
        aicc = self.fitted_model.aicc() if callable(self.fitted_model.aicc) else self.fitted_model.aicc
        print(f"AICc value: {aicc}")
        forecast_std = np.sqrt(aicc) / 10
        
        # Extract the predicted values from the dictionary format
        if isinstance(forecast[0], dict) and "forecast" in forecast[0]:
            # Extract the predicted_mean values from the list of dictionaries
            predicted_values = [item["forecast"] for item in forecast]
            forecast_upper = [val + std_error_factor * forecast_std for val in predicted_values]
            forecast_lower = [val - std_error_factor * forecast_std for val in predicted_values]
        else:
            # Original behavior for non-dictionary forecasts
            forecast_upper = forecast + std_error_factor * forecast_std
            forecast_lower = forecast - std_error_factor * forecast_std
            
        return forecast_lower, forecast_upper


@task(name="Extract BTC Prices", retries=2, retry_delay_seconds=5)
def extract_bitcoin_prices(tickers: str, period: str, interval: str) -> pd.DataFrame:
    """Extract BTC prices from Yahoo Finance"""
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        multi_level_index=False
    )
    return data


@task(name="Save BTC Price Data as CSV")
def load_btc(data: pd.DataFrame, path: str) -> None:
    """Save BTC price data to CSV"""
    data.to_csv(path_or_buf=path, index=True)
    return path


@task(name="Plot Time Series")
def plot_time_series(data, title='Bitcoin Price History', save_path=None):
    """Plot time series data"""
    plt.figure(figsize=(10, 7))
    plt.plot(data.index, data['Close'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return buf


@task(name="Plot Forecast")
def plot_forecast(train, test=None, forecast=None, forecast_lower=None, 
                 forecast_upper=None, future_dates=None, future_forecast=None,
                 future_lower=None, future_upper=None, 
                 title='Bitcoin Price Forecast', save_path=None):
    """Plot historical data and forecasts"""
    plt.figure(figsize=(10, 7))
    
    # Plot historical training data
    plt.plot(train.index, train['Close'], label='Training Data')
    
    # Plot test data if provided
    if test is not None:
        plt.plot(test.index, test['Close'], label='Actual Test Data')
    
    # Plot test forecast if provided
    if forecast is not None:
        # Extract predicted_mean values if forecast contains dictionaries
        if isinstance(forecast[0], dict) and "predicted_mean" in forecast[0]:
            forecast_values = [item["predicted_mean"] for item in forecast]
        else:
            forecast_values = forecast
            
        # Ensure forecast has same length as test data
        if len(forecast_values) != len(test.index):
            print(f"Warning: forecast length ({len(forecast_values)}) doesn't match test index length ({len(test.index)})")
            # Use only as many points as we have in test data
            forecast_values = forecast_values[:len(test.index)]
        plt.plot(test.index, forecast_values, label='Forecast', color='red')
        
        # Plot confidence intervals if provided
        if forecast_lower is not None and forecast_upper is not None:
            # Ensure same length
            forecast_lower = forecast_lower[:len(test.index)]
            forecast_upper = forecast_upper[:len(test.index)]
            plt.fill_between(test.index, forecast_lower, forecast_upper, 
                             color='red', alpha=0.1, 
                             label='95% Confidence Interval')
    
    # Plot future forecast if provided
    if future_forecast is not None and future_dates is not None:
        # Extract predicted_mean values if future_forecast contains dictionaries
        if isinstance(future_forecast[0], dict) and "forecast" in future_forecast[0]:
            future_forecast_values = [item["forecast"] for item in future_forecast]
        else:
            future_forecast_values = future_forecast
            
        # Ensure future_forecast has same length as future_dates
        if len(future_forecast_values) != len(future_dates):
            print(f"Warning: future_forecast length ({len(future_forecast_values)}) doesn't match future_dates length ({len(future_dates)})")
            if len(future_forecast_values) > len(future_dates):
                # Truncate future_forecast to match future_dates
                future_forecast_values = future_forecast_values[:len(future_dates)]
            else:
                # Truncate future_dates to match future_forecast
                future_dates = future_dates[:len(future_forecast_values)]
        
        plt.plot(future_dates, future_forecast_values, label='Future Forecast', 
                 color='green', linestyle='--')
        
        # Plot future confidence intervals if provided
        if future_lower is not None and future_upper is not None:
            # Ensure same length
            future_lower = future_lower[:len(future_dates)]
            future_upper = future_upper[:len(future_dates)]
            plt.fill_between(future_dates, future_lower, future_upper, 
                            color='green', alpha=0.1, 
                            label='Future 95% Confidence Interval')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return buf


@task(name="Train Time Series Model")
def train_model(data, train_size=0.8, seasonal=False, stepwise=True, 
               max_p=5, max_q=5):
    """Train ARIMA time series model"""
    # Prepare data for modeling
    btc_series = data[['Close']].copy()
    
    # Split data into training and testing sets
    split_point = len(btc_series) - int(len(btc_series) * (1 - train_size))
    train = btc_series.iloc[:split_point]
    test = btc_series.iloc[split_point:]
    
    print(f"Training set: {train.shape[0]} records")
    print(f"Testing set: {test.shape[0]} records")
    
    # Initialize and train the model (using auto_arima to find best parameters)
    model = BTCTimeSeriesModel(seasonal=seasonal, stepwise=stepwise, 
                              max_p=max_p, max_q=max_q)
    model.fit(train)
    
    return model, train, test


@task(name="Generate Forecasts")
def generate_forecasts(model, train, test, future_periods=20):
    """Generate test period and future forecasts"""
    # Generate test period forecast
    test_periods = len(test)
    print(f"Test periods: {test_periods}")
    
    # Wrap the integer in a list to match the updated type hint
    test_forecast = model.predict(None, [test_periods])
    
    print(f"Test data length: {len(test)}")
    print(f"Test forecast length: {len(test_forecast)}")
    
    # Check if lengths match
    if len(test) != len(test_forecast):
        print(f"WARNING: Mismatch between test data length ({len(test)}) and forecast length ({len(test_forecast)})")
        # If we don't have enough forecasts, extend the forecast to match test length
        if len(test_forecast) < len(test):
            additional_periods = len(test) - len(test_forecast)
            print(f"Extending forecast with additional {additional_periods} periods")
            additional_forecast = model.predict(None, [additional_periods])
            test_forecast.extend(additional_forecast)
        else:
            # Trim the forecast to match test length
            test_forecast = test_forecast[:len(test)]
    
    # Generate confidence intervals
    test_forecast_lower, test_forecast_upper = model.generate_confidence_intervals(test_forecast)
    
    # Extract predicted mean values for metric calculation
    test_forecast_values = [item["forecast"] for item in test_forecast]
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(test['Close'].values, test_forecast_values)
    rmse = np.sqrt(mean_squared_error(test['Close'].values, test_forecast_values))
    mape = np.mean(np.abs((test['Close'].values - np.array(test_forecast_values)) / test['Close'].values)) * 100
    
    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Generate future dates and forecasts
    last_date = test.index[-1]
    if isinstance(last_date, pd.Timestamp):
        # For timestamp-based indices
        freq = pd.infer_freq(test.index)
        if freq is None:
            # If frequency can't be inferred, use the average time delta
            deltas = np.diff(test.index.astype(np.int64)) / 10**9  # Convert to seconds
            avg_seconds = deltas.mean()
            future_dates = [last_date + timedelta(seconds=int((i+1)*avg_seconds)) for i in range(future_periods)]
        else:
            future_dates = pd.date_range(start=last_date, periods=future_periods+1, freq=freq)[1:]
    else:
        # For numeric indices
        if len(test.index) > 1:
            step = test.index[-1] - test.index[-2]
        else:
            step = 1
        future_dates = [last_date + (i+1)*step for i in range(future_periods)]
    
    # Generate future forecast - wrap in a list to match type hint
    future_forecast = model.predict(None, [future_periods])
    future_forecast_lower, future_forecast_upper = model.generate_confidence_intervals(future_forecast)
    
    return mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper, future_dates, future_forecast, future_forecast_lower, future_forecast_upper


@task(name="Log MLflow Results")
def log_mlflow_results(model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                       future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
                       mae, rmse, mape, btc_data, temp_dir):
    """Log model, parameters, metrics, and plots to MLflow"""
    # Plot original time series and log as artifact
    ts_plot_path = os.path.join(temp_dir, "btc_price_history.png")
    plot_time_series(btc_data, save_path=ts_plot_path)
    mlflow.log_artifact(ts_plot_path, "plots")
    
    # Log model parameters
    mlflow.log_param("p", model.p)
    mlflow.log_param("d", model.d)
    mlflow.log_param("q", model.q)
    mlflow.log_param("train_size", len(train))
    mlflow.log_param("test_size", len(test))
    
    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    
    # Plot test forecast and log as artifact
    test_forecast_plot_path = os.path.join(temp_dir, "btc_test_forecast.png")
    plot_forecast(
        train,
        test=test,
        forecast=test_forecast,
        forecast_lower=test_forecast_lower,
        forecast_upper=test_forecast_upper,
        title='Bitcoin Price Forecast - Test Period',
        save_path=test_forecast_plot_path
    )
    mlflow.log_artifact(test_forecast_plot_path, "plots")
    
    # Plot future forecast and log as artifact
    future_forecast_plot_path = os.path.join(temp_dir, "btc_future_forecast.png")
    plot_forecast(
        pd.concat([train, test]),
        future_dates=future_dates,
        future_forecast=future_forecast,
        future_lower=future_forecast_lower,
        future_upper=future_forecast_upper,
        title='Bitcoin Price Forecast - Future Periods',
        save_path=future_forecast_plot_path
    )
    mlflow.log_artifact(future_forecast_plot_path, "plots")
    
    try:
        # Create simple input example - use an integer for number of periods
        input_example = [5] 
        
        # Generate example output with proper wrapping
        example_output = model.predict(None, input_example)
        
        # Log the model with signature
        signature = infer_signature(input_example, example_output)
        
        # Log the model
        model_path = "btc_arima_model"
        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=model,
            input_example=input_example,
            signature=signature)
        
        # Return the model URI
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        return model_uri
    except Exception as e:
        # Fallback: Log model without signature if there's an error
        print(f"Error generating model signature: {e}")
        print("Logging model without signature...")
        
        model_path = "btc_arima_model"
        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=model)
        
        # Return the model URI
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        return model_uri
    

@flow(name="BTC Price Forecasting Pipeline")
def main_flow(
    tickers="BTC-USD",
    period="60d",
    interval="5m",
    train_size=0.8,
    future_periods=20,
    tracking_uri="http://127.0.0.1:8080",
    experiment_name="btc_price_forecasting"
):
    """Main Prefect flow that orchestrates the entire pipeline"""
    
    # Point to the running MLflow server
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Extract BTC prices
    print("Extracting BTC Prices")
    btc_data = extract_bitcoin_prices(tickers=tickers, period=period, interval=interval)

    # Use tempfile only for generating local plots
    with tempfile.TemporaryDirectory() as temp_dir:
        with mlflow.start_run(run_name='btc_forecast') as run:
            print(f"MLflow Run ID: {run.info.run_id}")

            print("Training Time Series Model")
            model, train, test = train_model(btc_data, train_size=train_size)

            print("Generating Forecasts")
            (mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper,
             future_dates, future_forecast, future_forecast_lower, future_forecast_upper) = generate_forecasts(model, train, test, future_periods)

            print("Logging Results to MLflow")
            model_uri = log_mlflow_results(
                model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
                mae, rmse, mape, btc_data, temp_dir
            )

            print(f"Model saved at: {model_uri}")

    print("Pipeline completed successfully!")
    print("To view results, open:", tracking_uri)

# Main Program
if __name__ == "__main__":
    main_flow()