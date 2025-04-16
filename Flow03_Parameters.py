#Run command
#Handle API failures
#Move key params to main flow

import pandas as pd
import yfinance as yf
from prefect import task, flow

#Extract (fecth from Yfinance API, and retry twice if it fails)
@task(
    name = "Extract BTC Prices",
    retries= 2,
    retry_delay_seconds=3
)

def extract_bitcoin_prices(tickers:str, period:str, interval:str) -> pd.DataFrame:
    data = yf.download(
        tickers = tickers,
        period = period,
        interval = interval
    )
    return data 

#Load 
@task(name= "Save BTC Price Data as CSV")
def load_btc (data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf=path, index= True)


#Prefect Flow
@flow(name= "BTC Price Pipeline")
def main_flow(
    tickers = "BTC-USD",
    period = "5d",
    interval = "1h",
    path = 'C:/TS Automation/Python BTC/BTC.csv'
):
    print('Extracting BTC Prices')
    df = extract_bitcoin_prices(
        tickers= tickers, period = period, interval= interval)
    print(f"Storing BTC Prices")
    df = load_btc(data=df, path= path)

#Main Program 
if __name__ == "__main__":
    main_flow(
        tickers= "BTC-USD",
        period= "5d",
        interval= "1h",
        path= 'C:/TS Automation/Python BTC/BTC.csv'
    )