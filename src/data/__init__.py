"""Data loading and preprocessing utilities."""

import os
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf
from omegaconf import DictConfig


class YahooDataLoader:
    """Yahoo Finance data loader with proper error handling and caching."""
    
    def __init__(self, config: DictConfig):
        """Initialize data loader.
        
        Args:
            config: Configuration containing data parameters
        """
        self.symbols = config.symbols
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.interval = config.get("interval", "1d")
        self.data_dir = config.get("data_dir", "data")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_data(self, symbol: str, force_download: bool = False) -> pd.DataFrame:
        """Download stock data for a single symbol.
        
        Args:
            symbol: Stock symbol
            force_download: Force re-download even if cached
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        cache_file = os.path.join(self.data_dir, f"{symbol}_{self.start_date}_{self.end_date}.csv")
        
        if os.path.exists(cache_file) and not force_download:
            print(f"Loading cached data for {symbol}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        print(f"Downloading data for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval
            )
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Cache the data
            data.to_csv(cache_file)
            print(f"Data cached for {symbol}")
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def download_all_data(self, force_download: bool = False) -> dict:
        """Download data for all symbols.
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            dict: Dictionary mapping symbols to their data
        """
        all_data = {}
        
        for symbol in self.symbols:
            data = self.download_data(symbol, force_download)
            if not data.empty:
                all_data[symbol] = data
            else:
                print(f"Skipping {symbol} due to download error")
        
        return all_data
    
    def get_combined_data(self, force_download: bool = False) -> pd.DataFrame:
        """Get combined data for all symbols with multi-index.
        
        Args:
            force_download: Force re-download even if cached
            
        Returns:
            pd.DataFrame: Combined data with symbol and date index
        """
        all_data = self.download_all_data(force_download)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_data = []
        for symbol, data in all_data.items():
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            combined_data.append(data_copy)
        
        combined_df = pd.concat(combined_data, ignore_index=False)
        combined_df = combined_df.set_index(['symbol', combined_df.index])
        
        return combined_df


class DataPreprocessor:
    """Data preprocessing utilities for time series data."""
    
    def __init__(self, config: DictConfig):
        """Initialize preprocessor.
        
        Args:
            config: Configuration containing preprocessing parameters
        """
        self.lookback_window = config.get("lookback_window", 60)
        self.scaler_type = config.get("scaler_type", "minmax")
    
    def create_sequences(self, data: pd.DataFrame, target_col: str = "close") -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        import numpy as np
        
        values = data[target_col].values.reshape(-1, 1)
        
        X, y = [], []
        for i in range(self.lookback_window, len(values)):
            X.append(values[i-self.lookback_window:i, 0])
            y.append(values[i, 0])
        
        return np.array(X), np.array(y)
    
    def time_based_split(self, data: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data based on time periods to avoid look-ahead bias.
        
        Args:
            data: Input data
            train_end: End date for training set
            val_end: End date for validation set
            
        Returns:
            Tuple of (train, val, test) dataframes
        """
        train_data = data[data.index <= train_end]
        val_data = data[(data.index > train_end) & (data.index <= val_end)]
        test_data = data[data.index > val_end]
        
        return train_data, val_data, test_data
    
    def add_returns(self, data: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Add return columns to the data.
        
        Args:
            data: Input data
            price_col: Price column name
            
        Returns:
            Data with return columns added
        """
        data = data.copy()
        
        # Daily returns
        data['daily_return'] = data[price_col].pct_change()
        
        # Log returns
        data['log_return'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Forward returns for labels
        data['forward_return_1d'] = data[price_col].shift(-1) / data[price_col] - 1
        data['forward_return_5d'] = data[price_col].shift(-5) / data[price_col] - 1
        
        return data
