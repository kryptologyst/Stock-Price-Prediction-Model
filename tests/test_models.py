"""Test suite for the stock price prediction model."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
from omegaconf import OmegaConf

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, EarlyStopping
from data import YahooDataLoader, DataPreprocessor
from features import TechnicalFeatureEngineer
from models import LSTMModel, LSTMTrainer
from backtest import VectorBTBacktest


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are deterministic
        np.random.seed(42)
        val1 = np.random.random()
        set_seed(42)
        val2 = np.random.random()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Test improvement
        assert not early_stopping(0.1)  # Good loss
        assert not early_stopping(0.05)  # Better loss
        assert not early_stopping(0.03)  # Even better
        
        # Test no improvement
        assert not early_stopping(0.04)  # Worse but within patience
        assert not early_stopping(0.05)  # Still worse
        assert not early_stopping(0.06)  # Still worse
        
        # Test early stopping
        assert early_stopping(0.07)  # Should trigger early stopping


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_yahoo_data_loader_init(self):
        """Test YahooDataLoader initialization."""
        config = OmegaConf.create({
            'symbols': ['AAPL'],
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'data_dir': 'test_data'
        })
        
        loader = YahooDataLoader(config)
        assert loader.symbols == ['AAPL']
        assert loader.start_date == '2020-01-01'
        assert loader.end_date == '2020-12-31'
    
    @patch('yfinance.Ticker')
    def test_download_data(self, mock_ticker):
        """Test data downloading."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        mock_ticker.return_value.history.return_value = mock_data
        
        config = OmegaConf.create({
            'symbols': ['AAPL'],
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'data_dir': 'test_data'
        })
        
        loader = YahooDataLoader(config)
        data = loader.download_data('AAPL')
        
        assert not data.empty
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_data_preprocessor(self):
        """Test data preprocessing."""
        config = OmegaConf.create({
            'lookback_window': 5,
            'scaler_type': 'minmax'
        })
        
        preprocessor = DataPreprocessor(config)
        
        # Create test data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        X, y = preprocessor.create_sequences(data)
        
        assert len(X) == len(y)
        assert X.shape[1] == 5  # lookback_window
        assert len(X) == len(data) - 5  # Should be len(data) - lookback_window
    
    def test_time_based_split(self):
        """Test time-based data splitting."""
        config = OmegaConf.create({'lookback_window': 5})
        preprocessor = DataPreprocessor(config)
        
        # Create test data with date index
        dates = pd.date_range('2020-01-01', periods=100)
        data = pd.DataFrame({
            'close': np.random.randn(100)
        }, index=dates)
        
        train, val, test = preprocessor.time_based_split(
            data, '2020-03-01', '2020-06-01'
        )
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(data)


class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_technical_feature_engineer_init(self):
        """Test TechnicalFeatureEngineer initialization."""
        config = OmegaConf.create({
            'lookback_window': 60,
            'indicators': ['sma_20', 'rsi_14']
        })
        
        engineer = TechnicalFeatureEngineer(config)
        assert engineer.lookback_window == 60
        assert 'sma_20' in engineer.indicators
    
    def test_add_sma(self):
        """Test SMA calculation."""
        config = OmegaConf.create({'lookback_window': 60})
        engineer = TechnicalFeatureEngineer(config)
        
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        result = engineer.add_sma(data, 5)
        assert 'sma_5' in result.columns
        assert not result['sma_5'].isna().all()
    
    def test_add_rsi(self):
        """Test RSI calculation."""
        config = OmegaConf.create({'lookback_window': 60})
        engineer = TechnicalFeatureEngineer(config)
        
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        result = engineer.add_rsi(data, 5)
        assert 'rsi_5' in result.columns
        assert not result['rsi_5'].isna().all()
    
    def test_engineer_features(self):
        """Test complete feature engineering."""
        config = OmegaConf.create({
            'lookback_window': 60,
            'indicators': ['sma_20', 'rsi_14']
        })
        
        engineer = TechnicalFeatureEngineer(config)
        
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = engineer.engineer_features(data)
        assert len(result.columns) > 5  # Should have more than OHLCV


class TestModels:
    """Test model functionality."""
    
    def test_lstm_model_init(self):
        """Test LSTM model initialization."""
        model = LSTMModel(input_size=1, hidden_sizes=[50, 50], dropout=0.2)
        assert model is not None
        assert len(model.lstm_layers) == 2
    
    def test_lstm_model_forward(self):
        """Test LSTM model forward pass."""
        model = LSTMModel(input_size=1, hidden_sizes=[50], dropout=0.2)
        
        # Create test input
        x = torch.randn(32, 60, 1)  # batch_size=32, seq_len=60, input_size=1
        
        output = model(x)
        assert output.shape == (32, 1)  # batch_size=32, output_size=1
    
    def test_lstm_trainer_init(self):
        """Test LSTM trainer initialization."""
        config = OmegaConf.create({
            'units': [50, 50],
            'dropout': 0.2,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 5
        })
        
        trainer = LSTMTrainer(config)
        assert trainer is not None
        assert trainer.device is not None
    
    def test_prepare_data(self):
        """Test data preparation for training."""
        config = OmegaConf.create({
            'units': [50],
            'dropout': 0.2,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 5
        })
        
        trainer = LSTMTrainer(config)
        
        # Create test data
        X_train = np.random.randn(100, 60, 1)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 60, 1)
        y_val = np.random.randn(20)
        
        train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_val, y_val)
        
        assert train_loader is not None
        assert val_loader is not None


class TestBacktesting:
    """Test backtesting functionality."""
    
    def test_vectorbt_backtest_init(self):
        """Test VectorBTBacktest initialization."""
        config = OmegaConf.create({
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'slippage': 0.0005,
            'benchmark': 'SPY',
            'rebalance_frequency': 'daily'
        })
        
        backtest = VectorBTBacktest(config)
        assert backtest.initial_capital == 100000
        assert backtest.transaction_cost == 0.001
    
    def test_create_signals(self):
        """Test signal creation."""
        config = OmegaConf.create({
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'slippage': 0.0005
        })
        
        backtest = VectorBTBacktest(config)
        
        # Create test data
        predictions = np.array([100, 102, 98, 105, 103])
        prices = pd.Series([100, 101, 99, 104, 102])
        
        signals = backtest.create_signals(predictions, prices, threshold=0.02)
        
        assert len(signals) == len(predictions)
        assert all(signal in [-1, 0, 1] for signal in signals)


if __name__ == "__main__":
    pytest.main([__file__])
