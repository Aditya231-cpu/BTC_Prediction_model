import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

class BTCDataLoader:
    def __init__(self):
        self.ticker = "BTC-USD"
    
    def fetch_data(self, period="10y"):
        """
        can BTC historical data using yfinance
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        try:
            print(f" Downloading {period} of BTC-USD data")
            btc = yf.download(self.ticker, period=period, progress=True)
            btc = btc[['Close']].rename(columns={'Close': 'price'})
            btc = btc.dropna()
            print(f" Successfully fetched {len(btc)} days of BTC data")
            print(f" Date range: {btc.index[0].date()} to {btc.index[-1].date()}")
            return btc
        except Exception as e:
            print(f" Error fetching data: {e}")
            return None
    
    def save_data(self, data, filename='data/btc_data.csv'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data.to_csv(filename)
        print(f" Data saved to {filename}")

# Test the data loader
if __name__ == "__main__":
    loader = BTCDataLoader()
    btc_data = loader.fetch_data(period="2y")  # Start with 2 years for testing
    if btc_data is not None:
        loader.save_data(btc_data)
        print("Sample data:")
        print(btc_data.head())
        print(btc_data.tail())