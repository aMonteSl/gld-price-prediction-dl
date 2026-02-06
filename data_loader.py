"""
Data loading module for GLD historical price data using yfinance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class GLDDataLoader:
    """Load and preprocess GLD (Gold ETF) historical data."""
    
    def __init__(self, ticker='GLD', start_date=None, end_date=None):
        """
        Initialize the data loader.
        
        Args:
            ticker: Stock ticker symbol (default: 'GLD')
            start_date: Start date for data (default: 5 years ago)
            end_date: End date for data (default: today)
        """
        self.ticker = ticker
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365*5)
            
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def load_data(self):
        """Download historical data from yfinance."""
        print(f"Downloading {self.ticker} data from {self.start_date} to {self.end_date}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.ticker}")
            
        print(f"Downloaded {len(self.data)} records")
        return self.data
    
    def get_data(self):
        """Get the loaded data, loading if necessary."""
        if self.data is None:
            self.load_data()
        return self.data
    
    def compute_returns(self, horizon=1):
        """
        Compute future returns at given horizon.
        
        Args:
            horizon: Number of days ahead to compute returns
            
        Returns:
            Series of future returns
        """
        data = self.get_data()
        returns = (data['Close'].shift(-horizon) - data['Close']) / data['Close']
        return returns
    
    def compute_signals(self, horizon=1, threshold=0.0):
        """
        Compute buy/no-buy signals based on future returns.
        
        Args:
            horizon: Number of days ahead
            threshold: Return threshold for buy signal (default: 0.0)
            
        Returns:
            Series of binary signals (1=buy, 0=no-buy)
        """
        returns = self.compute_returns(horizon)
        signals = (returns > threshold).astype(int)
        return signals
