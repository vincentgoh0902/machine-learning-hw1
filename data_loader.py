import yfinance as yf
import pandas as pd

def load_data(ticker="^GSPC", start_date="2021-01-01", end_date="2025-12-31"):
    """
    Fetches historical stock market data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol. Default is "^GSPC" for S&P 500.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the stock data.
    """
    # yf.download fetches the data. For single tickers, it returns standard OHLCV columns.
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # yfinance sometimes returns a MultiIndex on the columns if there are multiple tickers
    # or if the structure of the API changes. We check and drop the second level if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    # Forward-fill any NaN values to handle missing data smoothly (e.g., trading holidays etc.)
    data = data.ffill()
    
    return data
