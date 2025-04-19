import yfinance as yf
from prefect import task, flow
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import save_to_db, db_connect
from sqlalchemy import create_engine, insert, MetaData, Table

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
    return save_to_db(data, table_name="btc_usd_prices")


@flow(name="BTC Data Extraction Flow", flow_run_name = "BTC Data Extraction")
def extraction_flow(
    tickers="BTC-USD",
    start = '2025-02-19',
    end = '2025-04-19',
    interval= "5m"
):
    """Flow to extract BTC price data and save it"""
    # Extract data
    data = extract_bitcoin_prices(tickers=tickers, start=start, end=end, interval=interval)
    # Save data
    save_btc_data(data)
    return True

def get_latest_timestamp_from_db():
    """Query PostgreSQL for the latest BTC timestamp"""
    engine = create_engine(db_connect())
    conn = engine.connect()
    try:
        result = conn.execute("SELECT MAX(datetime) FROM btc_usd_prices")
        latest_timestamp = result.scalar()
    except Exception as e:
        print(f"Error fetching latest timestamp: {e}")
        latest_timestamp = None
    finally:
        conn.close()
    return latest_timestamp
    
    
# Schedule the extraction flow to run every hour
def schedule_extraction_flow():
    """Schedule the extraction flow to run hourly and log it"""
    start_time = get_latest_timestamp_from_db()
    if start_time is None:
        start_time = datetime.now() - timedelta(days=60)
    end_time = datetime.now()

    print(f"[Scheduler] Extracting from {start_time} to {end_time}")
    try:
        data = extract_bitcoin_prices(tickers="BTC-USD", start=start_time, end=end_time, interval="5m")
        saved_data = save_btc_data(data)
        records_saved = len(saved_data)
        log_extraction_run(start_time, end_time, status="success", records_saved=records_saved)
        print(f"Extraction successful: {records_saved} records saved")
    except Exception as e:
        print(f"Extraction failed: {e}")
        log_extraction_run(start_time, end_time, status="failure", error_message=str(e))


if __name__ == "__main__":
    # Run main extraction flow once at startup
    extraction_flow()

    # Schedule the flow to run every hour
    scheduler = BlockingScheduler()
    scheduler.add_job(schedule_extraction_flow, 'interval', hours=1)
    print("Scheduler started. Running every hour...")
    scheduler.start()