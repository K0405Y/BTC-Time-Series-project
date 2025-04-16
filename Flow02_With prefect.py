#Recreate the extraction script

import pandas as pd
import yfinance as yf
from prefect import task, flow

#Extract
@task
def extract_bitcoin_prices() -> pd.DataFrame: 
    data = yf.download(
        tickers= "BTC-USD",
        period= "20d",
        interval = "5m"
    )
    return data 

#Transform
# @task
# def transform (data: pd.DataFrame) -> pd.DataFrame:
#     return data

#Load
@task
def load_btc (data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf=path, index= True)

#prefect flow
@flow
def main_flow(log_prints = True):
    print(">>>Extracting BTC prices")
    df = extract_bitcoin_prices()
    # print(">>> Doing the transformation")
    # df = transform()
    print(">>> Storing BTC prices")
    load_btc(
        data = df, 
        path = 'C:/TS Automation/Python BTC/BTC.csv'
    )

#Main program

if __name__ == "__main__":
    main_flow()