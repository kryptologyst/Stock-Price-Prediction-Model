# Stock Price Prediction Model

A research-grade stock price prediction system using LSTM neural networks and advanced technical indicators. This project demonstrates state-of-the-art techniques in financial time series forecasting with proper backtesting and evaluation.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH AND EDUCATIONAL PROJECT ONLY**

- This model is **NOT financial advice**
- Past performance does **NOT** guarantee future results
- All predictions may be **inaccurate**
- **DO NOT** use this for actual trading decisions
- Always consult with qualified financial advisors
- This software is provided "as is" without warranty

## Features

- **Modern Architecture**: LSTM neural networks with proper regularization
- **Technical Indicators**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Realistic Backtesting**: VectorBT integration with transaction costs and slippage
- **Time-Series Aware**: Proper time-based splits to avoid look-ahead bias
- **Interactive Demo**: Streamlit web application for model exploration
- **Reproducible**: Deterministic seeding and comprehensive logging
- **Production Ready**: Clean code structure with type hints and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Stock-Price-Prediction-Model.git
cd Stock-Price-Prediction-Model

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Training

```bash
# Train models for all configured symbols
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom_config.yaml
```

### Demo Application

```bash
# Launch interactive demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   ├── backtest/          # Backtesting framework
│   ├── risk/              # Risk management utilities
│   └── utils/             # Common utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── data/                  # Data storage
├── models/                # Trained model checkpoints
└── logs/                  # Experiment logs
```

## Configuration

The system uses Hydra/OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/data/yahoo_finance.yaml`: Data source configuration
- `configs/features/technical_indicators.yaml`: Feature engineering
- `configs/models/lstm_baseline.yaml`: Model architecture
- `configs/backtest/vectorbt_config.yaml`: Backtesting parameters

## Data Sources

The system supports multiple data sources:

- **Yahoo Finance**: Primary data source via `yfinance`
- **Alpha Vantage**: Alternative API (configured but not active)
- **Custom CSV**: Local data files

### Data Schema

```python
# Expected data format
{
    'open': float,      # Opening price
    'high': float,      # High price
    'low': float,       # Low price
    'close': float,     # Closing price
    'volume': int,      # Trading volume
    'adj_close': float  # Adjusted closing price
}
```

## Models

### LSTM Neural Network

- **Architecture**: Multi-layer LSTM with dropout regularization
- **Input**: 60-day lookback window with technical indicators
- **Output**: Next-day price prediction
- **Training**: Adam optimizer with early stopping
- **Regularization**: Dropout and L2 regularization

### Technical Indicators

The system includes 20+ technical indicators:

- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume SMA, Volume Ratio
- **Price**: High-Low Ratio, Price Position

## Backtesting

### Realistic Simulation

- **Transaction Costs**: 0.1% per trade
- **Slippage**: 0.05% market impact
- **Benchmark Comparison**: S&P 500 (SPY)
- **Risk Metrics**: Sharpe ratio, max drawdown, Calmar ratio

### Evaluation Metrics

**Machine Learning Metrics:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

**Trading Metrics:**
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Hit Rate
- Profit Factor

## Usage Examples

### Basic Training

```python
from omegaconf import OmegaConf
from src.data import YahooDataLoader
from src.models import LSTMTrainer

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Load data
data_loader = YahooDataLoader(config.data)
data = data_loader.download_data("AAPL")

# Train model
trainer = LSTMTrainer(config.models.lstm)
trainer.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = trainer.predict(X_test)
```

### Custom Feature Engineering

```python
from src.features import TechnicalFeatureEngineer

# Create custom feature set
feature_engineer = TechnicalFeatureEngineer(config.features)
data_with_features = feature_engineer.engineer_features(data)

# Get feature columns
feature_columns = feature_engineer.get_feature_columns(data_with_features)
```

### Backtesting

```python
from src.backtest import VectorBTBacktest

# Initialize backtesting system
backtest = VectorBTBacktest(config.backtest)

# Create trading signals
signals = backtest.create_signals(predictions, prices)

# Run backtest
results = backtest.run_backtest(prices, signals)

# Generate report
report = backtest.generate_report(results)
print(report)
```

## Development

### Code Quality

The project follows modern Python development practices:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Pytest test suite

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Performance Considerations

### Hardware Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 8GB+ for large datasets
- **GPU**: CUDA-compatible GPU for faster training (optional)
- **Storage**: 1GB+ for data and model storage

### Optimization Tips

- Use GPU acceleration for LSTM training
- Enable mixed precision training for large models
- Implement data caching for repeated experiments
- Use parallel processing for feature engineering

## Limitations and Risks

### Model Limitations

- **Market Regime Changes**: Models trained on historical data may not adapt to new market conditions
- **Overfitting**: Complex models may memorize training data
- **Data Quality**: Predictions depend on input data quality
- **Liquidity**: Model assumes sufficient market liquidity

### Risk Factors

- **Model Risk**: Predictions may be systematically wrong
- **Data Risk**: Input data may contain errors or biases
- **Implementation Risk**: Code bugs may affect results
- **Market Risk**: Unpredictable market events

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stock_price_prediction,
  title={Stock Price Prediction Model},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Stock-Price-Prediction-Model}
}
```

## Acknowledgments

- Yahoo Finance for providing free market data
- VectorBT team for the excellent backtesting framework
- PyTorch team for the deep learning framework
- Streamlit team for the web application framework

---

**Remember: This is research software. Not financial advice.**
# Stock-Price-Prediction-Model
