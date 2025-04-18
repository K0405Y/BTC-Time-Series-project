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
from typing import Union, List
import yfinance as yf
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from prefect import task, flow

# Suppress warnings
warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]


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
        
        if self.p is None or self.q is None:
            # Find best parameters with auto_arima
            self.model = self.find_best_params(train_data)
            self.fitted_model = self.model
        else:
            # Use user-specified parameters
            self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
            self.fitted_model = self.model.fit()
        
        return self
    
    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Union[int, pd.DataFrame, List[Union[int, pd.DataFrame]]]) -> Union[np.ndarray, pd.Series, List[Union[np.ndarray, pd.Series]]]:
        if isinstance(model_input, int):
            n_periods = model_input
            if hasattr(self.fitted_model, "predict") and "n_periods" in self.fitted_model.predict.__code__.co_varnames:
                # pmdarima-style predict
                forecast = self.fitted_model.predict(n_periods=n_periods)
            else:
                # statsmodels-style predict
                start = len(self.train_index)
                end = start + n_periods - 1
                forecast = self.fitted_model.predict(start=start, end=end)
            return forecast
        else:
            n_periods = len(model_input)
            forecast = self.fitted_model.predict(n_periods=n_periods)
            forecast = pd.Series(forecast, index=model_input.index)
            return forecast  
        
    def generate_confidence_intervals(self, forecast, std_error_factor=1.96):
        """Generate 95% confidence intervals for the forecast"""
        # Check if aicc is a method or float
        aicc = self.fitted_model.aicc() if callable(self.fitted_model.aicc) else self.fitted_model.aicc
        print(f"AICc value: {aicc}")
        forecast_std = np.sqrt(aicc) / 10
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
    plt.figure(figsize=(14, 7))
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
    plt.figure(figsize=(14, 7))
    
    # Plot historical training data
    plt.plot(train.index, train['Close'], label='Training Data')
    
    # Plot test data if provided
    if test is not None:
        plt.plot(test.index, test['Close'], label='Actual Test Data')
    
    # Plot test forecast if provided
    if forecast is not None:
        plt.plot(test.index, forecast, label='Forecast', color='red')
        
        # Plot confidence intervals if provided
        if forecast_lower is not None and forecast_upper is not None:
            plt.fill_between(test.index, forecast_lower, forecast_upper, 
                             color='red', alpha=0.1, 
                             label='95% Confidence Interval')
    
    # Plot future forecast if provided
    if future_forecast is not None and future_dates is not None:
        plt.plot(future_dates, future_forecast, label='Future Forecast', 
                 color='green', linestyle='--')
        
        # Plot future confidence intervals if provided
        if future_lower is not None and future_upper is not None:
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
    test_forecast = model.predict(None, test)
    
    # Generate confidence intervals
    test_forecast_lower, test_forecast_upper = model.generate_confidence_intervals(test_forecast)
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(test, test_forecast)
    rmse = np.sqrt(mean_squared_error(test, test_forecast))
    mape = np.mean(np.abs((test['Close'] - test_forecast) / test['Close'])) * 100
    
    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Generate future forecast
    full_data = pd.concat([train, test])
    model.fit(full_data)  # Refit on full data
    
    # Create future dates
    last_date = full_data.index[-1]
    time_delta = full_data.index[1] - full_data.index[0]  # Determine the time interval
    future_dates = pd.date_range(start=last_date + time_delta, periods=future_periods)
    
    # Generate future forecast
    future_forecast = model.predict(None, future_periods)
    
    # Generate confidence intervals for future forecast
    future_forecast_lower, future_forecast_upper = model.generate_confidence_intervals(future_forecast)
    
    return (test_forecast, test_forecast_lower, test_forecast_upper, 
           future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
           mae, rmse, mape)


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
    
    input_example = btc_data[['Close']].iloc[:5]  # Example input for the model

    # Generate signature
    example_model = model.fit(input_example)
    example_output = example_model.predict(None, input_example)
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
            (test_forecast, test_forecast_lower, test_forecast_upper,
             future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
             mae, rmse, mape) = generate_forecasts(model, train, test, future_periods)

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