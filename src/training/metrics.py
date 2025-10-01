"""
Trading performance metrics for model evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

class TradingMetrics:
    """Calculate trading performance metrics."""

    def __init__(self):
        pass

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0

    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        cumulative = cumulative / cumulative[0]  # Normalize to start at 1

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown) if len(drawdown) > 0 else 0.0

    def calculate_calmar_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0

        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        max_dd = abs(self.calculate_max_drawdown(returns))

        return annual_return / max_dd if max_dd > 0 else float('inf')

    def calculate_hit_ratio(self, returns: np.ndarray) -> float:
        """Calculate hit ratio (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0

        positive_returns = np.sum(returns > 0)
        return positive_returns / len(returns)

    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(returns) == 0:
            return 0.0

        profits = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))

        return profits / losses if losses > 0 else float('inf')

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, confidence_level * 100)

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0

        var_threshold = self.calculate_var(returns, confidence_level)
        tail_losses = returns[returns <= var_threshold]

        return np.mean(tail_losses) if len(tail_losses) > 0 else 0.0

    def calculate_information_ratio(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate Information ratio vs benchmark."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Calculate tracking error
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        # Annual information ratio
        annual_excess = np.mean(excess_returns) * 252
        annual_tracking_error = tracking_error * np.sqrt(252)

        return annual_excess / annual_tracking_error

    def calculate_turnover(self, positions: np.ndarray) -> float:
        """Calculate portfolio turnover."""
        if len(positions) < 2:
            return 0.0

        # Calculate absolute position changes
        position_changes = np.abs(np.diff(positions))

        # Average turnover
        return np.mean(position_changes)

    def calculate_trade_statistics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive trade statistics."""
        if len(returns) == 0:
            return {}

        # Basic statistics
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)
        downside_volatility = np.std(returns[returns < 0]) * np.sqrt(252)

        # Performance metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd = self.calculate_max_drawdown(returns)
        calmar = self.calculate_calmar_ratio(returns)

        # Trade metrics
        hit_ratio = self.calculate_hit_ratio(returns)
        profit_factor = self.calculate_profit_factor(returns)

        # Risk metrics
        var_95 = self.calculate_var(returns, 0.05)
        cvar_95 = self.calculate_expected_shortfall(returns, 0.05)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'hit_ratio': hit_ratio,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_volatility': downside_volatility
        }

    def calculate_rolling_sharpe(self, returns: np.ndarray, window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < window:
            return pd.Series(index=range(len(returns)), dtype=float)

        rolling_sharpe = []

        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            sharpe = self.calculate_sharpe_ratio(window_returns)
            rolling_sharpe.append(sharpe)

        return pd.Series(rolling_sharpe, index=range(window, len(returns) + 1))

    def calculate_rolling_max_drawdown(self, returns: np.ndarray, window: int = 252) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        if len(returns) < window:
            return pd.Series(index=range(len(returns)), dtype=float)

        rolling_dd = []

        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            dd = self.calculate_max_drawdown(window_returns)
            rolling_dd.append(dd)

        return pd.Series(rolling_dd, index=range(window, len(returns) + 1))

    def calculate_performance_attribution(self, returns: np.ndarray,
                                        benchmark_returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance attribution vs benchmark."""

        # Alpha and beta
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {'alpha': 0.0, 'beta': 0.0, 'correlation': 0.0}

        # Linear regression for alpha/beta
        slope, intercept, r_value, _, _ = stats.linregress(benchmark_returns, returns)

        # Annualize alpha
        annual_alpha = intercept * 252

        return {
            'alpha': annual_alpha,
            'beta': slope,
            'correlation': r_value ** 2
        }

    def calculate_risk_adjusted_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted metrics."""

        stats = self.calculate_trade_statistics(returns)

        # Additional risk metrics
        if len(returns) > 0:
            # Skewness and kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            # Tail risk measures
            tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5))

            stats.update({
                'skewness': skewness,
                'kurtosis': kurtosis,
                'tail_ratio': tail_ratio
            })

        return stats

    def create_performance_report(self, returns: np.ndarray,
                                benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """Create comprehensive performance report."""

        # Basic statistics
        basic_stats = self.calculate_trade_statistics(returns)

        # Risk-adjusted metrics
        risk_metrics = self.calculate_risk_adjusted_metrics(returns)

        # Benchmark comparison (if provided)
        benchmark_comparison = {}
        if benchmark_returns is not None:
            benchmark_comparison = self.calculate_performance_attribution(returns, benchmark_returns)

        # Rolling metrics
        rolling_sharpe = self.calculate_rolling_sharpe(returns)
        rolling_dd = self.calculate_rolling_max_drawdown(returns)

        report = {
            'basic_statistics': basic_stats,
            'risk_metrics': risk_metrics,
            'benchmark_comparison': benchmark_comparison,
            'rolling_metrics': {
                'sharpe': rolling_sharpe.to_dict() if len(rolling_sharpe) > 0 else {},
                'max_drawdown': rolling_dd.to_dict() if len(rolling_dd) > 0 else {}
            },
            'summary': {
                'total_trades': len(returns),
                'win_rate': basic_stats.get('hit_ratio', 0),
                'sharpe_ratio': basic_stats.get('sharpe_ratio', 0),
                'max_drawdown': basic_stats.get('max_drawdown', 0),
                'calmar_ratio': basic_stats.get('calmar_ratio', 0)
            }
        }

        return report
