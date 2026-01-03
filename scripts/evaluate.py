#!/usr/bin/env python3
"""Evaluation script for comparing different models."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import set_seed, get_device
from data import YahooDataLoader, DataPreprocessor
from features import TechnicalFeatureEngineer
from models import LSTMTrainer
from models.tree_models import XGBoostModel, LightGBMModel
from backtest import VectorBTBacktest


def evaluate_model(model_name: str, trainer, X_test: np.ndarray, y_test: np.ndarray, 
                  test_prices: pd.Series, config) -> Dict[str, Any]:
    """Evaluate a single model and return results.
    
    Args:
        model_name: Name of the model
        trainer: Trained model trainer
        X_test: Test features
        y_test: Test targets
        test_prices: Test price series
        config: Configuration
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {model_name}...")
    
    # Make predictions
    predictions = trainer.predict(X_test)
    
    # Calculate ML metrics
    ml_metrics = trainer.evaluate(X_test, y_test)
    
    # Create trading signals
    backtest_system = VectorBTBacktest(config.backtest)
    signals = backtest_system.create_signals(predictions.flatten(), test_prices)
    
    # Run backtest
    backtest_results = backtest_system.run_backtest(test_prices, signals)
    
    # Combine results
    results = {
        'model_name': model_name,
        'ml_metrics': ml_metrics,
        'backtest_results': backtest_results,
        'predictions': predictions,
        'signals': signals
    }
    
    return results


def compare_models(symbol: str = "AAPL") -> None:
    """Compare different models on a single symbol.
    
    Args:
        symbol: Stock symbol to evaluate
    """
    # Load configuration
    config = OmegaConf.load("configs/config.yaml")
    
    # Set seed for reproducibility
    set_seed(config.experiment.seed)
    
    print(f"Comparing models on {symbol}")
    print("=" * 50)
    
    # Load data
    data_loader = YahooDataLoader(config.data)
    data = data_loader.download_data(symbol)
    
    if data.empty:
        print(f"No data available for {symbol}")
        return
    
    # Preprocess data
    preprocessor = DataPreprocessor(config.features)
    data = preprocessor.add_returns(data)
    
    # Engineer features
    feature_engineer = TechnicalFeatureEngineer(config.features)
    features_df, feature_columns = feature_engineer.prepare_features(data)
    data_with_features = pd.concat([data, features_df], axis=1)
    data_with_features = data_with_features.dropna()
    
    # Time-based split
    train_data, val_data, test_data = preprocessor.time_based_split(
        data_with_features,
        config.data.train_end_date,
        config.data.val_end_date
    )
    
    # Create sequences
    X_train, y_train = preprocessor.create_sequences(train_data, "close")
    X_val, y_val = preprocessor.create_sequences(val_data, "close")
    X_test, y_test = preprocessor.create_sequences(test_data, "close")
    
    # Reshape for LSTM
    X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Prepare test prices
    test_prices = test_data['close'].iloc[config.features.lookback_window:]
    test_prices = test_prices.reset_index(drop=True)
    
    # Train and evaluate models
    models_to_evaluate = []
    
    # LSTM Model
    print("Training LSTM model...")
    lstm_trainer = LSTMTrainer(config.models.lstm)
    lstm_trainer.train(X_train_lstm, y_train, X_val_lstm, y_val)
    models_to_evaluate.append(("LSTM", lstm_trainer, X_test_lstm, y_test))
    
    # XGBoost Model
    print("Training XGBoost model...")
    xgb_trainer = XGBoostModel(config.models.xgboost)
    xgb_trainer.train(X_train_lstm, y_train, X_val_lstm, y_val)
    models_to_evaluate.append(("XGBoost", xgb_trainer, X_test_lstm, y_test))
    
    # LightGBM Model
    print("Training LightGBM model...")
    lgb_trainer = LightGBMModel(config.models.lightgbm)
    lgb_trainer.train(X_train_lstm, y_train, X_val_lstm, y_val)
    models_to_evaluate.append(("LightGBM", lgb_trainer, X_test_lstm, y_test))
    
    # Evaluate all models
    results = []
    for model_name, trainer, X_test_model, y_test_model in models_to_evaluate:
        result = evaluate_model(model_name, trainer, X_test_model, y_test_model, test_prices, config)
        results.append(result)
    
    # Create comparison report
    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    
    # ML Metrics Comparison
    print("\nMachine Learning Metrics:")
    print("-" * 50)
    ml_df = pd.DataFrame([{
        'Model': r['model_name'],
        'RMSE': r['ml_metrics']['rmse'],
        'MAE': r['ml_metrics']['mae'],
        'R²': r['ml_metrics']['r2'],
        'MAPE': r['ml_metrics']['mape']
    } for r in results])
    
    print(ml_df.to_string(index=False, float_format='%.4f'))
    
    # Trading Metrics Comparison
    print("\nTrading Performance Metrics:")
    print("-" * 50)
    trading_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Total Return': f"{r['backtest_results']['total_return']:.2%}",
        'Sharpe Ratio': f"{r['backtest_results']['sharpe_ratio']:.3f}",
        'Max Drawdown': f"{r['backtest_results']['max_drawdown']:.2%}",
        'Hit Rate': f"{r['backtest_results']['hit_rate']:.2%}",
        'Total Trades': r['backtest_results']['total_trades']
    } for r in results])
    
    print(trading_df.to_string(index=False))
    
    # Create comparison plots
    create_comparison_plots(results, test_prices, symbol)
    
    # Save results
    save_results(results, symbol)


def create_comparison_plots(results: List[Dict[str, Any]], test_prices: pd.Series, symbol: str) -> None:
    """Create comparison plots for all models.
    
    Args:
        results: List of model results
        test_prices: Test price series
        symbol: Stock symbol
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predictions vs Actual
    axes[0, 0].plot(test_prices.values[:100], label='Actual', alpha=0.7, linewidth=2)
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, result in enumerate(results):
        predictions = result['predictions'][:100]
        axes[0, 0].plot(predictions, label=f"{result['model_name']}", 
                       alpha=0.7, color=colors[i % len(colors)])
    
    axes[0, 0].set_title(f'{symbol} - Predictions vs Actual')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Equity Curves
    for i, result in enumerate(results):
        portfolio = result['backtest_results']['portfolio']
        equity_curve = portfolio.value()
        axes[0, 1].plot(equity_curve.index, equity_curve.values, 
                       label=f"{result['model_name']}", 
                       color=colors[i % len(colors)], linewidth=2)
    
    axes[0, 1].set_title('Equity Curves Comparison')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Portfolio Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Comparison
    for i, result in enumerate(results):
        portfolio = result['backtest_results']['portfolio']
        drawdown_curve = portfolio.drawdowns.drawdown_series
        axes[1, 0].fill_between(drawdown_curve.index, drawdown_curve.values, 0, 
                               alpha=0.3, label=f"{result['model_name']}",
                               color=colors[i % len(colors)])
    
    axes[1, 0].set_title('Drawdown Comparison')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Drawdown')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Metrics Comparison (Bar chart)
    metrics = ['RMSE', 'MAE', 'R²', 'Sharpe Ratio']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, result in enumerate(results):
        values = [
            result['ml_metrics']['rmse'],
            result['ml_metrics']['mae'],
            result['ml_metrics']['r2'],
            result['backtest_results']['sharpe_ratio']
        ]
        axes[1, 1].bar(x + i * width, values, width, 
                      label=result['model_name'], 
                      color=colors[i % len(colors)], alpha=0.7)
    
    axes[1, 1].set_title('Metrics Comparison')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"assets/model_comparison_{symbol}.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to: {plot_path}")


def save_results(results: List[Dict[str, Any]], symbol: str) -> None:
    """Save evaluation results to files.
    
    Args:
        results: List of model results
        symbol: Stock symbol
    """
    # Create results directory
    os.makedirs("assets", exist_ok=True)
    
    # Save detailed results
    results_data = []
    for result in results:
        row = {
            'Model': result['model_name'],
            'RMSE': result['ml_metrics']['rmse'],
            'MAE': result['ml_metrics']['mae'],
            'R2': result['ml_metrics']['r2'],
            'MAPE': result['ml_metrics']['mape'],
            'Total_Return': result['backtest_results']['total_return'],
            'Sharpe_Ratio': result['backtest_results']['sharpe_ratio'],
            'Max_Drawdown': result['backtest_results']['max_drawdown'],
            'Hit_Rate': result['backtest_results']['hit_rate'],
            'Total_Trades': result['backtest_results']['total_trades']
        }
        results_data.append(row)
    
    # Save to CSV
    results_df = pd.DataFrame(results_data)
    csv_path = f"assets/evaluation_results_{symbol}.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")


def main():
    """Main evaluation function."""
    print("Stock Price Prediction Model Evaluation")
    print("=" * 50)
    
    # Evaluate on different symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        try:
            compare_models(symbol)
            print(f"\nCompleted evaluation for {symbol}")
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
            continue
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
