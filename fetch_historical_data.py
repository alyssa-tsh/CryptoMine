import requests
import pandas as pd
from datetime import datetime
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API
cg = CoinGeckoAPI()

def fetch_ohlc_data(crypto_id, currency="usd", days=30):
    """
    Fetch OHLC (Open, High, Low, Close) data for a given cryptocurrency.
    
    :param crypto_id: CoinGecko ID of the cryptocurrency (e.g., "bitcoin").
    :param currency: Quote currency (default: USD).
    :param days: Number of days of historical data.
    :return: Pandas DataFrame with OHLC data.
    """
    try:
        ohlc = cg.get_coin_ohlc_by_id(id=crypto_id, vs_currency=currency, days=days)
        if not ohlc:
            print(f"No data found for {crypto_id}")
            return None

        # Convert data to DataFrame
        df = pd.DataFrame(ohlc, columns=["timestamp", "open", "high", "low", "close"])
        
        # Convert timestamp to datetime and extract day and hour
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["day"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        
        # Drop timestamp and reorder columns
        df.drop(columns=["timestamp"], inplace=True)
        df = df[["datetime", "day", "hour", "open", "high", "low", "close"]]
        
        # Sort by datetime in descending order
        df.sort_values(by="hour", ascending=False, inplace=True)
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {crypto_id}: {e}")
        return None

def fetch_portfolio_ohlc(portfolio, days=30):
    """
    Fetch OHLC data for a portfolio of cryptocurrencies and save each as a CSV file.
    
    :param portfolio: List of cryptocurrency IDs.
    :param days: Number of days of historical data.
    """
    for crypto in portfolio:
        print(f"Fetching OHLC data for {crypto}...")
        df = fetch_ohlc_data(crypto, days=days)
        
        if df is not None:
            filename = f"{crypto}_ohlc_data.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}\n")
        else:
            print(f"Skipping {crypto} due to an error.\n")

# Define your portfolio (CoinGecko IDs)
portfolio = ["bitcoin", "ethereum", "binancecoin", "solana"]

# Fetch and save OHLC data for the entire portfolio
fetch_portfolio_ohlc(portfolio, days=30)
