import yfinance as yf
from prefect import task, flow
from datetime import timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import save_to_db, db_connect, get_latest_timestamp
from sqlalchemy import create_engine, insert, MetaData


#function to log the extraction run
def log_extraction_run(start_time, end_time, status, records_saved=0, error_message=None):
    engine = create_engine(db_connect())
    metadata = MetaData()
    
    # Reflect existing table from the database
    metadata.reflect(bind=engine, only=['extraction_runs'])
    extraction_runs = metadata.tables['extraction_runs']

    run_time = datetime.now()

    with engine.begin() as conn:  # automatically handles commit/rollback
        conn.execute(
            insert(extraction_runs),
            {
                'run_time': run_time,
                'start_time': start_time,
                'end_time': end_time,
                'status': status,
                'records_saved': records_saved,
                'error_message': error_message
            }
        )

@task(name="Extract BTC Prices", retries=2, retry_delay_seconds=5)
def extract_bitcoin_prices(tickers, start, end, interval = '5m'):
    """Extract BTC prices from Yahoo Finance"""
    print(f"Extracting {tickers} data for period: {start} - {end}, interval: {interval}")
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        multi_level_index=False
    )

@task(name="Save BTC Price Data")
def save_btc_data(data):
    """Save BTC price data to database"""   
    #rename the columns to match the database schema
    data = data.reset_index()
    data = data.rename(columns={"Datetime": "intervals"})
    data.columns = [col.lower() for col in data.columns]
    save_to_db(data, table_name="btc_usd_prices") 
    return True

@flow(name = "Main BTC Data Extraction Flow", log_prints = True, flow_run_name = "BTC Data Extraction")
def extraction_flow(
    tickers="BTC-USD",
    start = '2025-02-19',
    end = '2025-04-19',
    interval= "5m"
):
    """Flow to extract BTC price data and save it to the database"""
    # Extract data
    data = extract_bitcoin_prices(tickers=tickers, start=start, end=end, interval=interval)
    # Save data
    save_btc_data(data)
    return True
       
# Schedule the extraction flow to run every hour
@flow(name="Scheduled BTC Data Extraction Flow", log_prints=True, flow_run_name = "BTC Data Extraction Scheduler")
def schedule_extraction_flow():
    """Schedule the extraction flow to run hourly and log it"""
    start_time = get_latest_timestamp()
    if start_time is None:
        start_time = datetime.now() - timedelta(minutes=start_time.minute % 5, seconds=start_time.second, microseconds=start_time.microsecond)

    # Make end time to the last 5-minute mark
    now = datetime.now()
    end_time = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)

    print(f"[Scheduler] Extracting from {start_time} to {end_time}")
    try:
        data = extract_bitcoin_prices(tickers="BTC-USD", start=start_time, end=end_time, interval="5m")
        save_btc_data(data)
        records_saved = len(data['Close'])
        log_extraction_run(start_time, end_time, status="success", records_saved=records_saved)
        print(f"Extraction successful: {records_saved} records saved")
    except Exception as e:
        print(f"Extraction failed: {e}")
        log_extraction_run(start_time, end_time, status="failure", error_message=str(e))

if __name__ == "__main__":
    # Run main extraction flow once at startup
    # extraction_flow()
    # Schedule the flow to run every hour
    schedule_extraction_flow.serve(name = "BTC Data Extraction Job", cron = "0 * * * *")
    print("Scheduler started. Running every hour...")