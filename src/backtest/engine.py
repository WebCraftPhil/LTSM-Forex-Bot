"""
Backtesting engine using backtrader with risk management.
"""
import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.config import get_config, BacktestConfig
from ..utils.logging import get_logger, get_metrics_logger
from ..training.metrics import TradingMetrics
from .strategy import LSTMStrategy

logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)

class RiskManager(bt.Analyzer):
    """Risk management analyzer for backtrader."""

    def __init__(self):
        self.max_position_size = 0.02  # 2% of capital per trade
        self.max_positions = 5
        self.stop_loss = 0.02  # 2%
        self.take_profit = 0.04  # 4%
        self.trailing_stop = True
        self.circuit_breaker_drawdown = 0.15  # 15%

    def next(self):
        # Check circuit breaker
        if self.strategy.broker.getvalue() / self.strategy.initial_cash < (1 - self.circuit_breaker_drawdown):
            logger.warning("Circuit breaker triggered - excessive drawdown")
            self.strategy.circuit_breaker = True

    def get_analysis(self):
        return {
            'circuit_breaker_triggered': getattr(self.strategy, 'circuit_breaker', False),
            'max_positions': self.max_positions,
            'position_size_limit': self.max_position_size
        }

class PerformanceAnalyzer(bt.Analyzer):
    """Comprehensive performance analyzer."""

    def __init__(self):
        self.metrics = TradingMetrics()
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            # Calculate trade return
            pnl = trade.pnlcomm
            trade_return = pnl / self.strategy.initial_cash
            self.returns.append(trade_return)

    def get_analysis(self):
        if not self.returns:
            return {}

        returns_array = np.array(self.returns)
        return self.metrics.calculate_trade_statistics(returns_array)

class BacktestEngine:
    """Backtesting engine with risk management."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or get_config().backtest
        self.metrics = TradingMetrics()

    def run_backtest(self, data: Dict[str, pd.DataFrame], model_path: str,
                    report_path: Optional[str] = None) -> Dict:
        """Run backtest with LSTM model."""

        logger.info("Starting backtest")

        # Create cerebro engine
        cerebro = bt.Cerebro()

        # Set initial cash
        cerebro.broker.setcash(self.config.stake)

        # Set commission
        cerebro.broker.setcommission(commission=self.config.commission)

        # Add risk manager
        cerebro.addanalyzer(RiskManager, _name='risk')

        # Add performance analyzer
        cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')

        # Add strategy
        strategy = LSTMStrategy(model_path=model_path, config=self.config)
        cerebro.addstrategy(strategy)

        # Add data feeds
        for symbol, df in data.items():
            # Convert to backtrader format
            bt_data = self._convert_to_bt_format(df, symbol)
            cerebro.adddata(bt_data)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        logger.info(f"Running backtest with ${self.config.stake","} initial capital")

        # Run backtest
        results = cerebro.run()

        # Extract results
        backtest_results = self._extract_results(results)

        # Generate report
        if report_path:
            self._generate_report(backtest_results, report_path)

        logger.info("Backtest completed")
        return backtest_results

    def _convert_to_bt_format(self, df: pd.DataFrame, symbol: str) -> bt.feeds.PandasData:
        """Convert DataFrame to backtrader format."""

        # Ensure proper column names
        df = df.copy()
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        # Create backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # Not used
        )

        return data

    def _extract_results(self, results) -> Dict:
        """Extract results from backtrader run."""

        strategy = results[0]

        # Basic portfolio metrics
        final_value = strategy.broker.getvalue()
        total_return = (final_value - self.config.stake) / self.config.stake

        # Get analyzers
        risk_analysis = strategy.analyzers.risk.get_analysis()
        performance_analysis = strategy.analyzers.performance.get_analysis()
        sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        returns_analysis = strategy.analyzers.returns.get_analysis()
        trades_analysis = strategy.analyzers.trades.get_analysis()

        # Extract trade statistics
        trades_list = []
        if hasattr(strategy, 'trades'):
            for trade in strategy.trades:
                trades_list.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'pnl': trade.pnl,
                    'size': trade.size,
                    'entry_price': trade.pricein,
                    'exit_price': trade.priceout
                })

        # Calculate additional metrics
        if performance_analysis:
            trade_returns = np.array([t['pnl'] / self.config.stake for t in trades_list])
            portfolio_metrics = self.metrics.calculate_trade_statistics(trade_returns)
        else:
            portfolio_metrics = {}

        backtest_results = {
            'portfolio': {
                'initial_cash': self.config.stake,
                'final_value': final_value,
                'total_return': total_return,
                'pnl': final_value - self.config.stake
            },
            'risk_management': risk_analysis,
            'performance': portfolio_metrics,
            'sharpe_ratio': sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0,
            'max_drawdown': drawdown_analysis.get('max', {}).get('drawdown', 0) if drawdown_analysis else 0,
            'total_return': returns_analysis.get('rtot', 0) if returns_analysis else 0,
            'annual_return': returns_analysis.get('rnorm100', 0) if returns_analysis else 0,
            'trades': trades_list,
            'num_trades': len(trades_list),
            'winning_trades': len([t for t in trades_list if t['pnl'] > 0]),
            'losing_trades': len([t for t in trades_list if t['pnl'] <= 0])
        }

        return backtest_results

    def _generate_report(self, results: Dict, report_path: str):
        """Generate HTML report from backtest results."""

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots

            # Create equity curve
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Equity Curve', 'Drawdown', 'Trade Distribution', 'Rolling Sharpe', 'Monthly Returns', 'Trade P&L'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}],
                       [{"colspan": 2}, None]]
            )

            # Equity curve
            equity_curve = self._calculate_equity_curve(results['trades'])
            fig.add_trace(
                go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity'),
                row=1, col=1
            )

            # Drawdown
            drawdown_curve = self._calculate_drawdown_curve(equity_curve)
            fig.add_trace(
                go.Scatter(x=drawdown_curve.index, y=drawdown_curve.values, mode='lines', name='Drawdown', fill='tozeroy'),
                row=1, col=2
            )

            # Trade distribution
            if results['trades']:
                pnl_values = [t['pnl'] for t in results['trades']]
                fig.add_trace(
                    go.Histogram(x=pnl_values, name='Trade P&L'),
                    row=2, col=1
                )

            # Rolling Sharpe
            rolling_sharpe = self._calculate_rolling_sharpe(results['trades'])
            if not rolling_sharpe.empty:
                fig.add_trace(
                    go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode='lines', name='Rolling Sharpe'),
                    row=2, col=2
                )

            # Monthly returns
            monthly_returns = self._calculate_monthly_returns(results['trades'])
            if not monthly_returns.empty:
                fig.add_trace(
                    go.Bar(x=monthly_returns.index, y=monthly_returns.values, name='Monthly Returns'),
                    row=3, col=1
                )

            # Update layout
            fig.update_layout(
                title=f"Backtest Results - Total Return: {results['portfolio']['total_return']".2%"}",
                height=800,
                showlegend=False
            )

            # Save report
            report_dir = Path(report_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)

            fig.write_html(report_path)
            logger.info(f"Backtest report saved to {report_path}")

        except ImportError:
            logger.warning("Plotly not available, skipping HTML report generation")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def _calculate_equity_curve(self, trades: List[Dict]) -> pd.Series:
        """Calculate equity curve from trades."""

        if not trades:
            return pd.Series(dtype=float)

        # Sort trades by entry time
        sorted_trades = sorted(trades, key=lambda x: x['entry_time'])

        equity = [self.config.stake]

        for trade in sorted_trades:
            current_equity = equity[-1]
            new_equity = current_equity + trade['pnl']
            equity.append(new_equity)

        # Create time series
        dates = [pd.Timestamp('2020-01-01')]  # Start date
        if trades:
            dates.extend([pd.to_datetime(t['entry_time']) for t in sorted_trades])

        return pd.Series(equity, index=dates)

    def _calculate_drawdown_curve(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown curve."""

        if equity_curve.empty:
            return pd.Series(dtype=float)

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        return drawdown

    def _calculate_rolling_sharpe(self, trades: List[Dict], window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio."""

        if not trades:
            return pd.Series(dtype=float)

        # Extract trade returns
        trade_returns = [t['pnl'] / self.config.stake for t in trades]

        # Calculate rolling Sharpe
        returns_series = pd.Series(trade_returns)
        return self.metrics.calculate_rolling_sharpe(returns_series.values, window)

    def _calculate_monthly_returns(self, trades: List[Dict]) -> pd.Series:
        """Calculate monthly returns."""

        if not trades:
            return pd.Series(dtype=float)

        # Group trades by month
        monthly_pnl = {}

        for trade in trades:
            month_key = pd.to_datetime(trade['entry_time']).strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade['pnl']

        return pd.Series(monthly_pnl).sort_index()

def run_backtest(data: Dict[str, pd.DataFrame], model_path: str,
                config_path: Optional[str] = None, report_path: Optional[str] = None) -> Dict:
    """Run backtest with LSTM model."""

    # Load configuration
    if config_path:
        from ..utils.config import load_config
        config = load_config(config_path)
        backtest_config = config.backtest
    else:
        backtest_config = get_config().backtest

    # Create and run backtest engine
    engine = BacktestEngine(backtest_config)
    results = engine.run_backtest(data, model_path, report_path)

    # Log results
    metrics_logger.log_backtest_results(results)

    return results
