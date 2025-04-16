import pandas as pd
import yfinance as yf
from openpyxl import Workbook
data = yf.download(tickers = "BTC-USD", 
                   period = "5d", 
                   interval = "1h",
                   ignore_tz= True)

pd.DataFrame(data).to_csv("c:/TS Automation/Python BTC/BTC.csv") 

