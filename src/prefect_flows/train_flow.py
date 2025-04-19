# training.py - Simplified Model training flow
import os
import tempfile
from prefect import flow
import mlflow
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.train import (
    load_btc_data,
    train_model,
    generate_forecasts,
    log_mlflow_results
)

@flow(name="BTC Model Training Flow", flow_run_name="BTC Model Training", log_prints=True)
def training_flow(
    train_size=0.8,
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    experiment_name="btc_price_forecasting"
):
    """Flow to train BTC price forecasting model"""
    # Point to the running MLflow server
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Load data
    data = load_btc_data()
    
    # Use tempfile for MLflow artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        with mlflow.start_run(run_name='btc_forecast_training') as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Train model
            print("Training Time Series Model")
            model, train, test = train_model(data, train_size=train_size)

            print("Generating Forecasts")
            (mae, rmse, mape, test_forecast, test_forecast_lower, test_forecast_upper) = generate_forecasts(
                model, test
            )
            print(test_forecast[:5])
            print(test_forecast_lower[:5])
            print(test_forecast_upper[:5])

            print("Logging Results to MLflow")
            model_uri = log_mlflow_results(
                model, train, test, test_forecast, test_forecast_lower, test_forecast_upper,
                mae, rmse, mape, data, temp_dir
            )

            print(f"Model saved at: {model_uri}")

    print("Pipeline completed successfully!")
    print("To view results, open:", tracking_uri)


if __name__ == "__main__":
    training_flow()