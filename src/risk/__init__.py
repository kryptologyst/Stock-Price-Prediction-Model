"""Risk management utilities for portfolio and trading risk assessment."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import stats
from omegaconf import DictConfig


class RiskManager:
    """Risk management utilities for trading and portfolio management."""
    
    def __init__(self, config: DictConfig):
        """Initialize risk manager.
        
        Args:
            config: Configuration containing risk parameters
        """
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% max position
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.2)  # 20% max drawdown
        self.var_confidence = config.get("var_confidence", 0.95)
        self.lookback_window = config.get("lookback_window", 252)  # 1 year
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Return series
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Maximum drawdown as a percentage
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else 0
        
        return excess_returns.mean() / downside_std * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """Calculate Calmar ratio.
        
        Args:
            returns: Return series
            prices: Price series
            
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(prices))
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / max_dd
    
    def calculate_kelly_fraction(self, returns: pd.Series) -> float:
        """Calculate Kelly fraction for position sizing.
        
        Args:
            returns: Return series
            
        Returns:
            Kelly fraction
        """
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = abs(returns[returns < 0].mean())
        
        if avg_loss == 0:
            return 0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Cap Kelly fraction to reasonable limits
        return max(0, min(kelly, self.max_position_size))
    
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                               account_value: float) -> float:
        """Calculate position size based on signal strength and volatility.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            volatility: Asset volatility
            account_value: Total account value
            
        Returns:
            Position size in dollars
        """
        # Base position size
        base_size = account_value * self.max_position_size
        
        # Adjust for signal strength
        signal_adjusted_size = base_size * abs(signal_strength)
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = 1 / (1 + volatility)
        
        final_size = signal_adjusted_size * volatility_adjustment
        
        return min(final_size, account_value * self.max_position_size)
    
    def check_risk_limits(self, portfolio_value: float, max_drawdown: float, 
                         current_positions: Dict[str, float]) -> Dict[str, Any]:
        """Check if portfolio violates risk limits.
        
        Args:
            portfolio_value: Current portfolio value
            max_drawdown: Current maximum drawdown
            current_positions: Current position sizes
            
        Returns:
            Risk check results
        """
        violations = []
        
        # Check drawdown limit
        if max_drawdown > self.max_drawdown_limit:
            violations.append(f"Drawdown limit exceeded: {max_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
        
        # Check position size limits
        for symbol, position_size in current_positions.items():
            position_pct = position_size / portfolio_value
            if position_pct > self.max_position_size:
                violations.append(f"Position size limit exceeded for {symbol}: {position_pct:.2%} > {self.max_position_size:.2%}")
        
        # Check total exposure
        total_exposure = sum(abs(size) for size in current_positions.values())
        if total_exposure > portfolio_value:
            violations.append(f"Total exposure exceeds portfolio value: {total_exposure:.2f} > {portfolio_value:.2f}")
        
        return {
            'violations': violations,
            'risk_ok': len(violations) == 0,
            'max_drawdown': max_drawdown,
            'total_exposure': total_exposure
        }
    
    def calculate_portfolio_metrics(self, returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns: Portfolio return series
            prices: Portfolio price series
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns, prices),
            'max_drawdown': self.calculate_max_drawdown(prices),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'kelly_fraction': self.calculate_kelly_fraction(returns)
        }
        
        return metrics
    
    def generate_risk_report(self, returns: pd.Series, prices: pd.Series) -> str:
        """Generate a comprehensive risk report.
        
        Args:
            returns: Portfolio return series
            prices: Portfolio price series
            
        Returns:
            Formatted risk report
        """
        metrics = self.calculate_portfolio_metrics(returns, prices)
        
        report = f"""
RISK MANAGEMENT REPORT
======================

Performance Metrics:
- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
- Sortino Ratio: {metrics['sortino_ratio']:.3f}
- Calmar Ratio: {metrics['calmar_ratio']:.3f}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}

Risk Metrics:
- Volatility (Annualized): {metrics['volatility']:.2%}
- VaR (95%): {metrics['var_95']:.2%}
- CVaR (95%): {metrics['cvar_95']:.2%}

Distribution Metrics:
- Skewness: {metrics['skewness']:.3f}
- Kurtosis: {metrics['kurtosis']:.3f}

Position Sizing:
- Kelly Fraction: {metrics['kelly_fraction']:.3f}
- Max Position Size: {self.max_position_size:.1%}
- Max Drawdown Limit: {self.max_drawdown_limit:.1%}
"""
        
        return report
