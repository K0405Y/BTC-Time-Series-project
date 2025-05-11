import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union, List, Dict, Any
import mlflow
import mlflow.pyfunc
import os
import sys
import warnings
from sqlalchemy import create_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import (
    plot_time_series,
    plot_forecast,
    calculate_forecast_metrics,
    log_mlflow_model,
    db_connect
)
# Suppress warnings
warnings.filterwarnings("ignore")

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
            List of predictions as dictionaries with 'forecast' key
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
                return [{"forecast": float(x)} for x in forecast]
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
    
def load_btc_data(table_name="btc_usd_prices"):
    """Load BTC data from database"""
    engine = create_engine(db_connect())
    data = pd.read_sql(f"SELECT * FROM {table_name} where intervals >= '2025-03-01'", engine, index_col="intervals")
    return data

def train_model(data, train_size=0.8, seasonal=False, stepwise=True, 
            max_p=5, max_q=5):
    """Train ARIMA time series model"""
    # Prepare data for modeling
    btc_series = data[['close']].copy()
    
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

def generate_forecasts(model, test):
    """Generate test period and future forecasts"""
    # Generate test period forecast
    test_periods = len(test)
    print(f"Test periods: {test_periods}")
    
    # Generate test forecast
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
    
    # Calculate accuracy metrics
    mae, rmse, mape = calculate_forecast_metrics(test, test_forecast)
    
    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    return (mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper)

def log_mlflow_results(model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                    mae, rmse, mape, full_data, temp_dir):
    """Log model, parameters, metrics, and plots to MLflow"""
    # Plot original time series and log as artifact
    ts_plot_path = os.path.join(temp_dir, "btc_price_history.png")
    plot_time_series(full_data, save_path=ts_plot_path)
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
    
    # Log the model
    model_uri = log_mlflow_model(model)
    
    return model_uri