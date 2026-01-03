"""Technical indicators and feature engineering for stock price prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from omegaconf import DictConfig


class TechnicalFeatureEngineer:
    """Technical indicator feature engineer for stock price prediction."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature engineer.
        
        Args:
            config: Configuration containing feature engineering parameters
        """
        self.lookback_window = config.get("lookback_window", 60)
        self.indicators = config.get("indicators", [])
    
    def add_sma(self, data: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
        """Add Simple Moving Average indicator.
        
        Args:
            data: Input data
            window: SMA window
            price_col: Price column name
            
        Returns:
            Data with SMA column added
        """
        data[f'sma_{window}'] = data[price_col].rolling(window=window).mean()
        return data
    
    def add_ema(self, data: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
        """Add Exponential Moving Average indicator.
        
        Args:
            data: Input data
            window: EMA window
            price_col: Price column name
            
        Returns:
            Data with EMA column added
        """
        data[f'ema_{window}'] = data[price_col].ewm(span=window).mean()
        return data
    
    def add_rsi(self, data: pd.DataFrame, window: int = 14, price_col: str = "close") -> pd.DataFrame:
        """Add Relative Strength Index indicator.
        
        Args:
            data: Input data
            window: RSI window
            price_col: Price column name
            
        Returns:
            Data with RSI column added
        """
        delta = data[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        return data
    
    def add_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "close") -> pd.DataFrame:
        """Add MACD indicator.
        
        Args:
            data: Input data
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal line EMA window
            price_col: Price column name
            
        Returns:
            Data with MACD columns added
        """
        ema_fast = data[price_col].ewm(span=fast).mean()
        ema_slow = data[price_col].ewm(span=slow).mean()
        
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=signal).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        return data
    
    def add_bollinger_bands(self, data: pd.DataFrame, window: int = 20, std_dev: float = 2.0, price_col: str = "close") -> pd.DataFrame:
        """Add Bollinger Bands indicator.
        
        Args:
            data: Input data
            window: SMA window
            std_dev: Standard deviation multiplier
            price_col: Price column name
            
        Returns:
            Data with Bollinger Bands columns added
        """
        sma = data[price_col].rolling(window=window).mean()
        std = data[price_col].rolling(window=window).std()
        
        data['bb_upper'] = sma + (std * std_dev)
        data['bb_lower'] = sma - (std * std_dev)
        data['bb_middle'] = sma
        data['bb_width'] = data['bb_upper'] - data['bb_lower']
        data['bb_position'] = (data[price_col] - data['bb_lower']) / data['bb_width']
        
        return data
    
    def add_atr(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average True Range indicator.
        
        Args:
            data: Input data
            window: ATR window
            
        Returns:
            Data with ATR column added
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data[f'atr_{window}'] = true_range.rolling(window=window).mean()
        
        return data
    
    def add_volume_indicators(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add volume-based indicators.
        
        Args:
            data: Input data
            window: Window for volume indicators
            
        Returns:
            Data with volume indicators added
        """
        data[f'volume_sma_{window}'] = data['volume'].rolling(window=window).mean()
        data['volume_ratio'] = data['volume'] / data[f'volume_sma_{window}']
        
        return data
    
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features.
        
        Args:
            data: Input data
            
        Returns:
            Data with price features added
        """
        # Price ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Price position within daily range
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volatility (rolling standard deviation of returns)
        data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured technical indicators.
        
        Args:
            data: Input data
            
        Returns:
            Data with all features added
        """
        data = data.copy()
        
        # Add basic price features
        data = self.add_price_features(data)
        
        # Apply configured indicators
        for indicator in self.indicators:
            if indicator == "sma_20":
                data = self.add_sma(data, 20)
            elif indicator == "sma_50":
                data = self.add_sma(data, 50)
            elif indicator == "ema_12":
                data = self.add_ema(data, 12)
            elif indicator == "ema_26":
                data = self.add_ema(data, 26)
            elif indicator == "rsi_14":
                data = self.add_rsi(data, 14)
            elif indicator == "macd":
                data = self.add_macd(data)
            elif indicator == "bollinger_bands":
                data = self.add_bollinger_bands(data)
            elif indicator == "atr_14":
                data = self.add_atr(data, 14)
            elif indicator == "volume_sma_20":
                data = self.add_volume_indicators(data, 20)
        
        return data
    
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding OHLCV and target).
        
        Args:
            data: Input data
            
        Returns:
            List of feature column names
        """
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in base_columns]
        return feature_columns
    
    def prepare_features(self, data: pd.DataFrame, target_col: str = "close") -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling.
        
        Args:
            data: Input data
            target_col: Target column name
            
        Returns:
            Tuple of (features_df, feature_columns)
        """
        # Engineer features
        data_with_features = self.engineer_features(data)
        
        # Get feature columns
        feature_columns = self.get_feature_columns(data_with_features)
        
        # Select only feature columns
        features_df = data_with_features[feature_columns]
        
        return features_df, feature_columns
