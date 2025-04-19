# training.py - Model training flow
import pandas as pd
import numpy as np
import os
import tempfile
from prefect import task, flow
from prefect.schedules import IntervalSchedule
from datetime import timedelta
import mlflow
import warnings
from src.modeling.train_model import BTCTimeSeriesModel
from src.modeling.utility import (
    plot_time_series,
    plot_forecast,
    calculate_forecast_metrics,
    generate_future_dates,
    log_mlflow_model,
    db_connect
)
from sqlalchemy import create_engine

# Suppress warnings
warnings.filterwarnings("ignore")


@task(name="Load BTC Data")
def load_btc_data(table_name="btc_usd_prices"):
    """Load BTC data from database"""
    engine = create_engine(db_connect())
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine, index_col='timestamp')
    return data


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
    
    # Generate future dates and forecasts
    last_date = test.index[-1]
    future_dates = generate_future_dates(last_date, periods=future_periods, interval='5min')
    
    # Generate future forecast
    future_forecast = model.predict(None, [future_periods])
    future_forecast_lower, future_forecast_upper = model.generate_confidence_intervals(future_forecast)
    
    return (mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper, 
            future_dates, future_forecast, future_forecast_lower, future_forecast_upper)


@task(name="Log MLflow Results")
def log_mlflow_results(model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                       future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
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
    
    # Log the model
    model_uri = log_mlflow_model(model)
    
    return model_uri


@flow(name="BTC Model Training Flow")
def training_flow(
    from_db=True,
    train_size=0.8,
    future_periods=20,
    tracking_uri= os.getenv("MLFLOW_TRACKING_URI"),
    experiment_name="btc_price_forecasting"
):
    """Flow to train BTC price forecasting model"""
    # Point to the running MLflow server
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load data
    data = load_btc_data(from_db=from_db)
    
    # Use tempfile for MLflow artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        with mlflow.start_run(run_name='btc_forecast_training') as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Train model

            print("Training Time Series Model")
            model, train, test = train_model(data, train_size=train_size)

            print("Generating Forecasts")
            (mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper,
             future_dates, future_forecast, future_forecast_lower, future_forecast_upper) = generate_forecasts(model, train, test, future_periods)

            print("Logging Results to MLflow")
            model_uri = log_mlflow_results(
                model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                future_dates, future_forecast, future_forecast_lower, future_forecast_upper,
                mae, rmse, mape, data, temp_dir
            )

            print(f"Model saved at: {model_uri}")

    print("Pipeline completed successfully!")
    print("To view results, open:", tracking_uri)