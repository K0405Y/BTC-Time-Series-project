# model.py - Contains the BTC time series model definition
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union, List, Dict, Any
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature


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