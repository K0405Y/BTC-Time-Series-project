# inference.py - Script for making Bitcoin price predictions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import tempfile
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import (
    plot_forecast,
    generate_future_dates,
    save_predictions_to_db,
    db_connect,
    get_latest_timestamp
)

def load_latest_model(model_uri=None):
    """
    Load the latest trained model from MLflow registry or a specific URI
    
    Args:
        model_uri: If provided, will load this specific model. 
                  If None, will try to load the latest model from production.
    
    Returns:
        Loaded model object
    """
    if model_uri is None:
        # Try to get the latest model from production stage
        try:
            client = mlflow.tracking.MlflowClient()
            latest_model = client.get_latest_versions("btc_arima_model")
            if latest_model:
                model_uri = f"models:/btc_arima_model/Production"
            else:
                # Fallback to latest model in registry
                latest_model = client.search_model_versions("name='btc_arima_model'")
                if latest_model:
                    # Sort by version number to get the latest
                    latest_model.sort(key=lambda x: int(x.version), reverse=True)
                    model_uri = f"models:/btc_arima_model/{latest_model[0].version}"
                else:
                    raise ValueError("No models found in registry")
        except Exception as e:
            print(f"Error loading model from registry: {e}")
            raise ValueError("Failed to load model from registry. Please provide a model_uri.")
    
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict_next_hour(model=None, model_uri=None, intervals="5min", save_to_db=True):
    """
    Predict Bitcoin prices for the next hour with 5-minute intervals
    
    Args:
        model: Pre-loaded model object. If None, will load using model_uri
        model_uri: URI to load model from if model is None
        intervals: Time interval for predictions (default "5min")
        save_to_db: Whether to save predictions to database
    
    Returns:
        Tuple of (future_dates, predictions)
    """
    # Load model if not provided
    if model is None:
        model = load_latest_model(model_uri)
    
    # Get the latest timestamp from the database
    latest_timestamp = get_latest_timestamp()
    
    if latest_timestamp is None:
        print("No data found in database. Using current time as the reference.")
        latest_timestamp = datetime.now()
    
    print(f"Latest timestamp: {latest_timestamp}")
    
    # Calculate how many intervals we need for 1 hour
    if intervals == "5min":
        num_periods = 12  # 12 * 5min = 60min = 1 hour
    elif intervals == "1min":
        num_periods = 60
    else:
        num_periods = int(60 / int(intervals.replace("min", "")))
    
    # Generate future dates
    future_dates = generate_future_dates(latest_timestamp, periods=num_periods, interval=intervals)
    
    # Make predictions
    print(f"Predicting {num_periods} periods ({intervals} intervals)")
    predictions = model.predict([num_periods])
    
    # Output preview
    print("\nPrediction Preview:")
    for i, (date, pred) in enumerate(zip(future_dates[:5], predictions[:5])):
        if isinstance(pred, dict) and "forecast" in pred:
            pred_value = pred["forecast"]
        else:
            pred_value = pred
        print(f"{date}: ${pred_value:.2f}")
    
    if len(future_dates) > 5:
        print("...")
        
    # Save predictions to database if requested
    if save_to_db:
        save_predictions_to_db(predictions, future_dates)
    
    return future_dates, predictions


def visualize_predictions(future_dates, predictions, show_plot=True, save_path=None):
    """
    Visualize the predictions for the next hour
    
    Args:
        future_dates: List of future dates for predictions
        predictions: List of predictions (can be dictionaries with "forecast" key)
        show_plot: Whether to show the plot
        save_path: Path to save the plot
    
    Returns:
        Path to saved plot if save_path is provided, None otherwise
    """
    plt.figure(figsize=(10, 6))
    
    # Extract values if predictions are in dictionary format
    if isinstance(predictions[0], dict) and "forecast" in predictions[0]:
        pred_values = [p["forecast"] for p in predictions]
    else:
        pred_values = predictions
    
    # Plot the predictions
    plt.plot(future_dates, pred_values, marker='o', linestyle='-', color='blue')
    
    # Format the plot
    plt.title('Bitcoin Price Forecast - Next Hour')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to show time only
    plt.gcf().autofmt_xdate()
    
    # Add value labels
    for i, (date, value) in enumerate(zip(future_dates, pred_values)):
        if i % 2 == 0:  # Only label every other point to avoid overcrowding
            plt.text(date, value, f'${value:.2f}', fontsize=9, 
                     ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_path if save_path else None


def run_inference(model_uri=None, intervals="5min", save_to_db=True, 
                  show_plot=True, save_plot=True):
    """
    Main function to run the inference process
    
    Args:
        model_uri: URI to load model from (if None, will load latest from registry)
        intervals: Time interval for predictions (default "5min")
        save_to_db: Whether to save predictions to database
        show_plot: Whether to show the plot
        save_plot: Whether to save the plot to disk
    
    Returns:
        Tuple of (future_dates, predictions, plot_path)
    """
    # Load the model
    model = load_latest_model(model_uri)
    
    # Make predictions
    future_dates, predictions = predict_next_hour(
        model=model,
        intervals=intervals,
        save_to_db=save_to_db
    )
    
    # Generate plot
    plot_path = None
    if show_plot or save_plot:
        # Create a temporary directory for the plot if needed
        if save_plot and not os.path.exists('predictions'):
            os.makedirs('predictions')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"predictions/btc_forecast_{timestamp}.png" if save_plot else None
        
        visualize_predictions(
            future_dates=future_dates,
            predictions=predictions,
            show_plot=show_plot,
            save_path=plot_path
        )
    
    return future_dates, predictions, plot_path


if __name__ == "__main__":
    # Set up MLflow tracking URI if available
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Run the inference
    future_dates, predictions, plot_path = run_inference(
        intervals="5min",
        save_to_db=True,
        show_plot=True,
        save_plot=True
    )
    
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Predicted {len(predictions)} intervals of 5 minutes")
    
    # Extract predicted values if in dictionary format
    if isinstance(predictions[0], dict) and "forecast" in predictions[0]:
        pred_values = [p["forecast"] for p in predictions]
    else:
        pred_values = predictions
    
    print(f"Average predicted price: ${np.mean(pred_values):.2f}")
    print(f"Min predicted price: ${min(pred_values):.2f}")
    print(f"Max predicted price: ${max(pred_values):.2f}")
    
    if plot_path:
        print(f"\nPrediction plot saved to: {plot_path}")