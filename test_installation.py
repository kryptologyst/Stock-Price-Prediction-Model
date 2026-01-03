#!/usr/bin/env python3
"""Simple test script to verify the installation works correctly."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils import set_seed, get_device
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from data import YahooDataLoader, DataPreprocessor
        print("✓ Data modules imported successfully")
    except ImportError as e:
        print(f"✗ Data modules import failed: {e}")
        return False
    
    try:
        from features import TechnicalFeatureEngineer
        print("✓ Features imported successfully")
    except ImportError as e:
        print(f"✗ Features import failed: {e}")
        return False
    
    try:
        from models import LSTMTrainer, XGBoostModel
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from backtest import VectorBTBacktest
        print("✓ Backtest imported successfully")
    except ImportError as e:
        print(f"✗ Backtest import failed: {e}")
        return False
    
    try:
        from risk import RiskManager
        print("✓ Risk management imported successfully")
    except ImportError as e:
        print(f"✗ Risk management import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/config.yaml")
        print("✓ Configuration loaded successfully")
        print(f"  - Symbols: {config.data.symbols}")
        print(f"  - Date range: {config.data.start_date} to {config.data.end_date}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from utils import get_device
        device = get_device()
        print(f"✓ Device detected: {device}")
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False

def test_seed_setting():
    """Test seed setting."""
    print("\nTesting seed setting...")
    
    try:
        from utils import set_seed
        set_seed(42)
        print("✓ Seed set successfully")
        return True
    except Exception as e:
        print(f"✗ Seed setting failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")
    
    try:
        import pandas as pd
        import numpy as np
        from omegaconf import OmegaConf
        from features import TechnicalFeatureEngineer
        
        # Create test data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Create config
        config = OmegaConf.create({
            'lookback_window': 60,
            'indicators': ['sma_20', 'rsi_14']
        })
        
        # Test feature engineering
        engineer = TechnicalFeatureEngineer(config)
        result = engineer.engineer_features(data)
        
        print(f"✓ Feature engineering successful")
        print(f"  - Input shape: {data.shape}")
        print(f"  - Output shape: {result.shape}")
        print(f"  - Features added: {len(result.columns) - len(data.columns)}")
        
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Stock Price Prediction Model - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_device_detection,
        test_seed_setting,
        test_feature_engineering
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Launch demo: streamlit run demo/app.py")
        print("3. Run evaluation: python scripts/evaluate.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
