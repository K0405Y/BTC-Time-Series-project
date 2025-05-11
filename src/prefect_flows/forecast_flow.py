import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.inference.inference import inference_pipeline
from prefect import flow

@flow(name="Scheduled BTC Price Forecast", log_prints=True)
def inference_flow(model_uri=None, forecast_periods=36):
    try:
        result = inference_pipeline(model_uri=model_uri, forecast_periods=forecast_periods)
        print("Inference flow completed successfully")
        return result
    except Exception as e:
        print(f"Inference flow failed: {e}")
        return None

if __name__ == "__main__":
    inference_flow.serve(
        name="BTC Prediction Job",
        cron="0 */12 * * *",
        global_limit=1
    )
