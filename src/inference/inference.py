import os
import sys
import mlflow
import pandas as pd
from datetime import datetime, timedelta
from mlflow.tracking import MlflowClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import (
    get_latest_timestamp, 
    save_all_predictions,
    generate_future_dates,
    db_connect
)
from sqlalchemy import create_engine, insert, MetaData, select

def log_prediction_run(start_time, end_time, status, records_predicted=0, error_message=None):
    engine = create_engine(db_connect())
    metadata = MetaData()
    metadata.reflect(bind=engine, only=['prediction_runs'])
    prediction_runs = metadata.tables['prediction_runs']

    run_time = datetime.now()

    with engine.begin() as conn:
        conn.execute(
            insert(prediction_runs),
            {
                'run_time': run_time,
                'start_time': start_time,
                'end_time': end_time,
                'status': status,
                'records_predicted': records_predicted,
                'error_message': error_message
            }
        )

def load_model_from_uri(model_uri):
    return mlflow.pyfunc.load_model(model_uri)

def get_latest_model_uri():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("btc_price_forecasting")
    if not experiment:
        raise ValueError("Experiment not found")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError("No successful runs found")
    return f"runs:/{runs[0].info.run_id}/btc_arima_model"

def predict(model, forecast_periods):
    predictions = model.predict([forecast_periods])
    forecast_values = [item["forecast"] for item in predictions] if isinstance(predictions[0], dict) else predictions
    btc_model = model._model_impl.python_model
    forecast_lower, forecast_upper = btc_model.generate_confidence_intervals(predictions)
    return forecast_values, forecast_lower, forecast_upper

def inference_pipeline(model_uri=None, forecast_periods=36):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    if not model_uri:
        model_uri = get_latest_model_uri()

    model = load_model_from_uri(model_uri)
    last_date = get_latest_timestamp(table_name='btc_usd_predictions')
    if last_date is None:
        now = datetime.now()
        last_date = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)
    else:
        last_date = pd.to_datetime(last_date)
        if last_date.tzinfo is None:
            last_date = last_date.tz_localize("UTC")
        last_date = last_date.astimezone(tz=None).replace(tzinfo=None)

    future_dates = generate_future_dates(last_date=last_date, periods=forecast_periods)
    start_time = min(future_dates) if not future_dates.empty else None
    end_time = max(future_dates) if not future_dates.empty else None

    engine = create_engine(db_connect())
    metadata = MetaData()
    metadata.reflect(bind=engine, only=['prediction_runs'])
    prediction_runs = metadata.tables['prediction_runs']

    with engine.connect() as conn:
        existing_run = conn.execute(
            select(prediction_runs).where(
                (prediction_runs.c.start_time == start_time) &
                (prediction_runs.c.end_time == end_time) &
                (prediction_runs.c.status == 'success')
            )
        ).fetchone()
        if existing_run:
            print("[Scheduler] Already processed. Skipping.")
            return None

    forecast_values, forecast_lower, forecast_upper = predict(model, forecast_periods)
    records_count = forecast_periods if forecast_values else 0
    if records_count > 0:
        save_all_predictions(forecast_values, forecast_lower, forecast_upper, future_dates)
        log_prediction_run(start_time, end_time, status="success", records_predicted=records_count)
    else:
        log_prediction_run(start_time, end_time, status="success", records_predicted=0)

    return {
        "forecast_dates": future_dates,
        "forecast_values": forecast_values,
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper
    }