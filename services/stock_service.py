import requests
import pandas as pd

from config.settings import ALPHA_VANTAGE_API_KEY

BASE_URL = "https://www.alphavantage.co/query"


def get_daily_stock_data(symbol):
    params = {          
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }


#data preprocessing
    #sending the request
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json() #convert the json responses into the dictionary

    time_series = data["Time Series (Daily)"] #extract the daily stock prices

    df = pd.DataFrame.from_dict(time_series, orient="index") #converting the dict into dataframe
    
    df.columns = [
        "Open", "High", "Low", "Close", "Volume"  #renaming the columns
    ]

    df = df.astype(float) #string to numeric

    df.index = pd.to_datetime(df.index) #index to datetime

    df.sort_index(inplace=True) #sort oldest to newest


    return df