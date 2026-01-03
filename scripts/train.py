#!/usr/bin/env python3
"""Main training script for stock price prediction model."""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import set_seed, get_device, create_directories, log_experiment_info
from data import YahooDataLoader, DataPreprocessor
from features import TechnicalFeatureEngineer
from models import LSTMTrainer
from backtest import VectorBTBacktest


def main():
    """Main training pipeline."""
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    
    # Set seed for reproducibility
    set_seed(config.experiment.seed)
    
    # Create directories
    create_directories(config.paths)
    
    print("=" * 60)
    print("STOCK PRICE PREDICTION MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Symbols: {config.data.symbols}")
    print(f"Date Range: {config.data.start_date} to {config.data.end_date}")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = YahooDataLoader(config.data)
    all_data = data_loader.download_all_data()
    
    if not all_data:
        print("Error: No data loaded. Exiting.")
        return
    
    print(f"Loaded data for {len(all_data)} symbols")
    
    # Process each symbol
    results = {}
    
    for symbol in all_data.keys():
        print(f"\n2. Processing {symbol}...")
        
        # Get data for this symbol
        data = all_data[symbol]
        
        # Add returns
        preprocessor = DataPreprocessor(config.features)
        data = preprocessor.add_returns(data)
        
        # Engineer features
        feature_engineer = TechnicalFeatureEngineer(config.features)
        features_df, feature_columns = feature_engineer.prepare_features(data)
        
        # Combine features with price data
        data_with_features = pd.concat([data, features_df], axis=1)
        data_with_features = data_with_features.dropna()
        
        print(f"Features created: {len(feature_columns)} features")
        print(f"Data shape: {data_with_features.shape}")
        
        # Time-based split (critical for avoiding look-ahead bias)
        train_data, val_data, test_data = preprocessor.time_based_split(
            data_with_features,
            config.data.train_end_date,
            config.data.val_end_date
        )
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        if len(train_data) < config.features.lookback_window:
            print(f"Insufficient training data for {symbol}. Skipping.")
            continue
        
        # Create sequences
        X_train, y_train = preprocessor.create_sequences(train_data, "close")
        X_val, y_val = preprocessor.create_sequences(val_data, "close")
        X_test, y_test = preprocessor.create_sequences(test_data, "close")
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        print(f"Training sequences: {X_train.shape}")
        
        # Train model
        print(f"\n3. Training LSTM model for {symbol}...")
        trainer = LSTMTrainer(config.models.lstm)
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        print(f"\n4. Evaluating model for {symbol}...")
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Make predictions
        predictions = trainer.predict(X_test)
        
        # Prepare data for backtesting
        test_prices = test_data['close'].iloc[config.features.lookback_window:]
        test_prices = test_prices.reset_index(drop=True)
        
        # Create trading signals
        backtest_system = VectorBTBacktest(config.backtest)
        signals = backtest_system.create_signals(predictions.flatten(), test_prices)
        
        # Run backtest
        print(f"\n5. Running backtest for {symbol}...")
        backtest_results = backtest_system.run_backtest(test_prices, signals)
        
        # Generate report
        report = backtest_system.generate_report(backtest_results)
        print(report)
        
        # Save results
        results[symbol] = {
            'metrics': metrics,
            'backtest_results': backtest_results,
            'predictions': predictions,
            'actual': y_test,
            'signals': signals,
            'history': history
        }
        
        # Save model
        model_path = os.path.join(config.paths.models_dir, f"lstm_{symbol}.pt")
        trainer.save_model(model_path)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training history
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{symbol} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot 2: Predictions vs Actual
        plt.subplot(2, 2, 2)
        plt.plot(y_test[:100], label='Actual', alpha=0.7)
        plt.plot(predictions[:100], label='Predicted', alpha=0.7)
        plt.title(f'{symbol} - Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Plot 3: Equity curve
        plt.subplot(2, 2, 3)
        equity_curve = backtest_system.create_equity_curve(backtest_results['portfolio'])
        plt.plot(equity_curve.index, equity_curve.values)
        plt.title(f'{symbol} - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        
        # Plot 4: Drawdown
        plt.subplot(2, 2, 4)
        drawdown_curve = backtest_system.create_drawdown_curve(backtest_results['portfolio'])
        plt.fill_between(drawdown_curve.index, drawdown_curve.values, 0, alpha=0.3, color='red')
        plt.title(f'{symbol} - Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(config.paths.assets_dir, f"results_{symbol}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    summary_metrics = {}
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  RMSE: {result['metrics']['rmse']:.4f}")
        print(f"  MAE: {result['metrics']['mae']:.4f}")
        print(f"  R²: {result['metrics']['r2']:.4f}")
        print(f"  Sharpe Ratio: {result['backtest_results']['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {result['backtest_results']['max_drawdown']:.2%}")
        print(f"  Total Return: {result['backtest_results']['total_return']:.2%}")
        
        summary_metrics[symbol] = {
            'rmse': result['metrics']['rmse'],
            'mae': result['metrics']['mae'],
            'r2': result['metrics']['r2'],
            'sharpe_ratio': result['backtest_results']['sharpe_ratio'],
            'max_drawdown': result['backtest_results']['max_drawdown'],
            'total_return': result['backtest_results']['total_return']
        }
    
    # Log experiment
    log_experiment_info(config, {
        'summary_metrics': summary_metrics,
        'num_symbols': len(results),
        'feature_columns': feature_columns
    })
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {config.paths.assets_dir}")
    print(f"Models saved to: {config.paths.models_dir}")


if __name__ == "__main__":
    main()
