#Recreate the extraction script

import pandas as pd
import yfinance as yf

def extract_bitcoin_prices() -> pd.DataFrame: 
    data = yf.download(
        tickers= "BTC-USD",
        period= "5d",
        interval = "1h", 
        ignore_tz = True
    )
    return data

#Transformation (No transformation in this instance)
def transform(data:pd.DataFrame) -> pd.DataFrame:
    return data

def load_bitcoin_prices(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path_or_buf= path, index= True)


if __name__ == "__main__":
    df = extract_bitcoin_prices()
    df = transform(df)
    load_bitcoin_prices(data = df, path= 'C:/TS Automation/Python BTC/BTC5.csv')