import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def plot_time_series(data, title='Bitcoin Price History', save_path=None):
    """Plot time series data"""
    plt.figure(figsize=(10, 7))
    plt.plot(data.index, data['close'])
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

def plot_forecast(train, test=None, forecast=None, forecast_lower=None, 
                 forecast_upper=None, title='Bitcoin Price Forecast', save_path=None):
    """Plot historical data and forecasts"""
    plt.figure(figsize=(10, 7))
    
    # Plot historical training data
    plt.plot(train.index, train['close'], label='Training Data')
    
    # Plot test data if provided
    if test is not None:
        plt.plot(test.index, test['close'], label='Actual Test Data')
    
    # Plot test forecast if provided
    if forecast is not None:
        # Extract predicted_mean values if forecast contains dictionaries
        if isinstance(forecast[0], dict) and "forecast" in forecast[0]:
            forecast_values = [item["forecast"] for item in forecast]
        elif isinstance(forecast[0], dict) and "forecast" in forecast[0]:
            forecast_values = [item["forecast"] for item in forecast]
        else:
            forecast_values = forecast
            
        # Ensure forecast has same length as test data
        if test is not None and len(forecast_values) != len(test.index):
            print(f"Warning: forecast length ({len(forecast_values)}) doesn't match test index length ({len(test.index)})")
            # Use only as many points as we have in test data
            forecast_values = forecast_values[:len(test.index)]
        
        if test is not None:
            plt.plot(test.index, forecast_values, label='Forecast', color='red')
        
        # Plot confidence intervals if provided
        if forecast_lower is not None and forecast_upper is not None and test is not None:
            # Ensure same length
            forecast_lower = forecast_lower[:len(test.index)]
            forecast_upper = forecast_upper[:len(test.index)]
            plt.fill_between(test.index, forecast_lower, forecast_upper, 
                             color='red', alpha=0.1, 
                             label='95% Confidence Interval')
        
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

def calculate_forecast_metrics(test_actual, test_forecast):
    """Calculate accuracy metrics for forecasts"""
    # Extract forecast values if in dictionary format
    if isinstance(test_forecast[0], dict) and "forecast" in test_forecast[0]:
        test_forecast_values = [item["forecast"] for item in test_forecast]
    else:
        test_forecast_values = test_forecast
    
    # Calculate metrics
    mae = mean_absolute_error(test_actual['close'].values, test_forecast_values)
    rmse = np.sqrt(mean_squared_error(test_actual['close'].values, test_forecast_values))
    mape = np.mean(np.abs((test_actual['close'].values - np.array(test_forecast_values)) / test_actual['close'].values)) * 100
    
    return mae, rmse, mape

def generate_future_dates(last_date, periods=12, interval='5min'):
    """Generate future dates based on the last available date"""
    if isinstance(last_date, pd.Timestamp) or isinstance(last_date, datetime):
        # For timestamp-based indices
        return pd.date_range(start=last_date, periods=periods + 1, freq=interval)[1:]
    elif isinstance(last_date, (int, float)):
        # For numeric indices
        return [last_date + (i + 1) for i in range(periods)]
    else:
        raise TypeError("Unsupported type for last_date. Must be datetime, pd.Timestamp, int, or float.")
    
def log_mlflow_model(model, input_example=None):
    """Log model to MLflow"""
    try:
        if input_example is None:
            input_example = [5]

        example_output = model.predict(None, input_example)
        signature = infer_signature(input_example, example_output)

        model_path = "btc_arima_model"

        # Dynamically resolve the absolute path to the 'src' directory
        current_dir = os.path.dirname(os.path.abspath(__file__))  # points to src/modeling
        code_dir = os.path.abspath(os.path.join(current_dir, ".."))  # points to src

        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=model,
            input_example=input_example,
            signature=signature,
            code_path=[code_dir]  
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        return model_uri

    except Exception as e:
        print(f"Error generating model signature: {e}")
        print("Logging model without signature...")

        mlflow.pyfunc.log_model(
            artifact_path=model_path,
            python_model=model,
            input_example=input_example,
            code_path=[code_dir]
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        return model_uri
    
# Database utilities
def db_connect():
    """Connect to PostgreSQL database"""
    load_dotenv()
    return os.getenv('POSTGRES_URL')

def save_to_db(data, table_name="btc_usd_prices"):
    """Save data to database"""
    engine = create_engine(db_connect())
    print(f"Data saved to {table_name} table")
    return data.to_sql(table_name, engine, if_exists='append', index=False)

def get_latest_timestamp(table_name="btc_usd_prices"):
    """Get the latest timestamp from database """
    engine = create_engine(db_connect())
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT MAX(intervals) FROM {table_name}"))
        latest_timestamp = result.scalar()
    return latest_timestamp

def save_all_predictions(forecast_values, forecast_lower, forecast_upper, future_dates):
    """Save predictions to database with upper and lower bounds"""
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame({
        'interval': future_dates,
        'forecasted_price': forecast_values,
        'forecast_lower': forecast_lower,
        'forecast_upper': forecast_upper
    })
    
    # Save to database
    table_name = "btc_usd_predictions"

    engine = create_engine(db_connect())
   # Now save the predictions
    pred_df.to_sql(table_name, engine, if_exists='append', index=False)
    
    print(f"Saved {len(pred_df)} predictions to {table_name} table")
    print(pred_df.head())
    
    # Disconnect from database
    engine.dispose()
    
    return True