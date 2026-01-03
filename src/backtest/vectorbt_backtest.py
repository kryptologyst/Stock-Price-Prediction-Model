"""Backtesting module using VectorBT for realistic trading simulation."""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig


class VectorBTBacktest:
    """VectorBT-based backtesting system with realistic transaction costs."""
    
    def __init__(self, config: DictConfig):
        """Initialize backtesting system.
        
        Args:
            config: Configuration containing backtesting parameters
        """
        self.initial_capital = config.get("initial_capital", 100000)
        self.transaction_cost = config.get("transaction_cost", 0.001)
        self.slippage = config.get("slippage", 0.0005)
        self.benchmark = config.get("benchmark", "SPY")
        self.rebalance_frequency = config.get("rebalance_frequency", "daily")
        
        # Set VectorBT settings
        vbt.settings.set_theme("dark")
        vbt.settings['array_wrapper']['freq'] = 'D'
    
    def prepare_prices(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Prepare price data for backtesting.
        
        Args:
            data: Stock data
            symbol: Stock symbol
            
        Returns:
            Price series for backtesting
        """
        if isinstance(data.index, pd.MultiIndex):
            # Multi-symbol data
            prices = data.loc[symbol, 'close']
        else:
            # Single symbol data
            prices = data['close']
        
        return prices.dropna()
    
    def create_signals(self, predictions: np.ndarray, prices: pd.Series, 
                      threshold: float = 0.02) -> pd.Series:
        """Create trading signals from predictions.
        
        Args:
            predictions: Model predictions
            prices: Price series
            threshold: Signal threshold (2% by default)
            
        Returns:
            Trading signals (-1, 0, 1)
        """
        # Calculate expected returns
        expected_returns = predictions - prices.shift(1)
        expected_returns = expected_returns / prices.shift(1)
        
        # Create signals based on threshold
        signals = pd.Series(0, index=prices.index)
        signals[expected_returns > threshold] = 1  # Buy
        signals[expected_returns < -threshold] = -1  # Sell
        
        return signals
    
    def run_backtest(self, prices: pd.Series, signals: pd.Series, 
                    benchmark_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run backtest with VectorBT.
        
        Args:
            prices: Price series
            signals: Trading signals
            benchmark_prices: Benchmark prices for comparison
            
        Returns:
            Backtest results
        """
        # Align signals with prices
        signals = signals.reindex(prices.index, fill_value=0)
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            prices,
            signals,
            init_cash=self.initial_capital,
            fees=self.transaction_cost,
            slippage=self.slippage,
            freq='D'
        )
        
        # Calculate metrics
        results = {
            'portfolio': portfolio,
            'total_return': portfolio.total_return(),
            'annualized_return': portfolio.annualized_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'sortino_ratio': portfolio.sortino_ratio(),
            'calmar_ratio': portfolio.calmar_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'volatility': portfolio.annualized_volatility(),
            'hit_rate': portfolio.trades.win_rate(),
            'profit_factor': portfolio.trades.profit_factor(),
            'total_trades': len(portfolio.trades.records_readable),
            'avg_trade_return': portfolio.trades.returns.mean(),
        }
        
        # Add benchmark comparison if available
        if benchmark_prices is not None:
            benchmark_portfolio = vbt.Portfolio.from_holding(
                benchmark_prices,
                init_cash=self.initial_capital,
                freq='D'
            )
            
            results['benchmark_return'] = benchmark_portfolio.total_return()
            results['excess_return'] = results['total_return'] - results['benchmark_return']
            results['information_ratio'] = results['excess_return'] / portfolio.returns.std()
        
        return results
    
    def create_equity_curve(self, portfolio: vbt.Portfolio) -> pd.Series:
        """Create equity curve from portfolio.
        
        Args:
            portfolio: VectorBT portfolio
            
        Returns:
            Equity curve
        """
        return portfolio.value()
    
    def create_drawdown_curve(self, portfolio: vbt.Portfolio) -> pd.Series:
        """Create drawdown curve from portfolio.
        
        Args:
            portfolio: VectorBT portfolio
            
        Returns:
            Drawdown curve
        """
        return portfolio.drawdowns.drawdown_series
    
    def get_trade_analysis(self, portfolio: vbt.Portfolio) -> pd.DataFrame:
        """Get detailed trade analysis.
        
        Args:
            portfolio: VectorBT portfolio
            
        Returns:
            Trade analysis DataFrame
        """
        trades = portfolio.trades.records_readable
        if len(trades) == 0:
            return pd.DataFrame()
        
        return trades
    
    def calculate_risk_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """Calculate additional risk metrics.
        
        Args:
            portfolio: VectorBT portfolio
            
        Returns:
            Risk metrics dictionary
        """
        returns = portfolio.returns
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio
        tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5))
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
        }
    
    def plot_results(self, portfolio: vbt.Portfolio, benchmark_portfolio: Optional[vbt.Portfolio] = None,
                    save_path: Optional[str] = None) -> None:
        """Plot backtest results.
        
        Args:
            portfolio: Main portfolio
            benchmark_portfolio: Benchmark portfolio for comparison
            save_path: Path to save plots
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        portfolio.value().vbt.plot(ax=axes[0, 0], title="Equity Curve")
        if benchmark_portfolio is not None:
            benchmark_portfolio.value().vbt.plot(ax=axes[0, 0], title="Equity Curve")
        axes[0, 0].legend(['Strategy', 'Benchmark'])
        
        # Drawdown
        portfolio.drawdowns.drawdown_series.vbt.plot(ax=axes[0, 1], title="Drawdown")
        
        # Returns distribution
        portfolio.returns.hist(bins=50, ax=axes[1, 0], title="Returns Distribution")
        
        # Rolling Sharpe ratio
        portfolio.rolling_sharpe_ratio().vbt.plot(ax=axes[1, 1], title="Rolling Sharpe Ratio")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive backtest report.
        
        Args:
            results: Backtest results
            
        Returns:
            Formatted report string
        """
        report = f"""
BACKTEST REPORT
===============

Strategy Performance:
- Total Return: {results['total_return']:.2%}
- Annualized Return: {results['annualized_return']:.2%}
- Volatility: {results['volatility']:.2%}
- Sharpe Ratio: {results['sharpe_ratio']:.3f}
- Sortino Ratio: {results['sortino_ratio']:.3f}
- Calmar Ratio: {results['calmar_ratio']:.3f}
- Maximum Drawdown: {results['max_drawdown']:.2%}

Trading Statistics:
- Total Trades: {results['total_trades']}
- Hit Rate: {results['hit_rate']:.2%}
- Profit Factor: {results['profit_factor']:.3f}
- Average Trade Return: {results['avg_trade_return']:.2%}

Risk Metrics:
- VaR (95%): {results.get('var_95', 'N/A'):.2%}
- CVaR (95%): {results.get('cvar_95', 'N/A'):.2%}
- Skewness: {results.get('skewness', 'N/A'):.3f}
- Kurtosis: {results.get('kurtosis', 'N/A'):.3f}
"""
        
        if 'benchmark_return' in results:
            report += f"""
Benchmark Comparison:
- Benchmark Return: {results['benchmark_return']:.2%}
- Excess Return: {results['excess_return']:.2%}
- Information Ratio: {results['information_ratio']:.3f}
"""
        
        return report
