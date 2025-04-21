import yfinance as yf
from prefect import task, flow
from datetime import timedelta, datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.modeling.utility import save_to_db, db_connect, get_latest_timestamp
from sqlalchemy import create_engine, insert, MetaData, select, update


# Function to log the extraction run
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
@flow(name="Scheduled BTC Data Extraction Flow", log_prints=True)
def schedule_extraction_flow():
    """Schedule the extraction flow to run hourly and log it"""
   
    try:
        # Get the latest timestamp from the database
        start_time = get_latest_timestamp()
        
        # Make end time to the last 5-minute mark
        now = datetime.now()
        end_time = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)
        
        if start_time is not None and start_time.tzinfo is not None:
            # If start_time has timezone info, make it naive by converting to UTC and removing tzinfo
            start_time = start_time.astimezone(tz=None).replace(tzinfo=None)
        
        # Check if start_time is None (first run) or if there's actually data to extract
        if start_time is None:
            start_time = end_time - timedelta(hours=1)  # Default to fetching last hour's data
        elif start_time >= end_time:
            print(f"[Scheduler] No new data to extract. Latest timestamp: {start_time}, End time: {end_time}")
            return  # Skip this run if there's no new data to extract
            
        print(f"[Scheduler] Extracting from {start_time} to {end_time}")
        
        # Create a lock based on the time range to prevent duplicate runs
        engine = create_engine(db_connect())
        metadata = MetaData()
        metadata.reflect(bind=engine, only=['extraction_runs'])
        extraction_runs = metadata.tables['extraction_runs']
        
        with engine.connect() as conn:
            # Check if this exact range has already been processed successfully
            existing_run = conn.execute(
                select(extraction_runs).where(
                    (extraction_runs.c.start_time == start_time) &
                    (extraction_runs.c.end_time == end_time) &
                    (extraction_runs.c.status == 'success')
                )
            ).fetchone()
            
            if existing_run:
                print(f"[Scheduler] This exact time range has already been processed successfully. Skipping.")
                return
        
        # Extract and save data
        data = extract_bitcoin_prices(tickers="BTC-USD", start=start_time, end=end_time, interval="5m")
        records_count = len(data) if not data.empty else 0
        
        if records_count > 0:
            save_btc_data(data)
            log_extraction_run(start_time, end_time, status="success", records_saved=records_count)
            print(f"Extraction successful: {records_count} records saved")
        else:
            print(f"No new records found for time range: {start_time} - {end_time}")
            log_extraction_run(start_time, end_time, status="success", records_saved=0)
            
    except Exception as e:
        print(f"Extraction failed: {e}")
        if 'start_time' in locals() and 'end_time' in locals():
            log_extraction_run(start_time, end_time, status="failure", error_message=str(e))
        else:
            # If error happened before start_time/end_time were defined
            log_extraction_run(None, None, status="failure", error_message=str(e))

if __name__ == "__main__":
    # Run main extraction flow once at startup
    # extraction_flow()
    # Schedule the flow to run every hour
    schedule_extraction_flow.serve(name="BTC Data Extraction Job", cron="0 * * * *", global_limit=1)
    print("Scheduler started. Running every hour...")