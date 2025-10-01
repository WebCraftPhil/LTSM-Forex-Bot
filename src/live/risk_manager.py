"""
Risk management system for live trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging

from ..utils.config import get_config
from ..utils.logging import get_logger, get_metrics_logger
from ..training.metrics import TradingMetrics
from .broker_base import Position, Order

logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)

class RiskManager:
    """Comprehensive risk management for trading."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.metrics = TradingMetrics()

        # Risk parameters
        self.max_position_size = 0.02  # 2% of portfolio per trade
        self.max_positions = 5  # Maximum concurrent positions
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_drawdown = 0.15  # 15% max drawdown
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.trailing_stop = True
        self.trailing_stop_pct = 0.01  # 1% trailing stop

        # State tracking
        self.portfolio_history = []
        self.daily_pnl = []
        self.positions = {}
        self.orders = {}
        self.initial_balance = 0
        self.peak_balance = 0
        self.daily_start_balance = 0
        self.last_reset = datetime.now()

        # Risk events
        self.risk_events = []

        # Callbacks
        self.on_risk_event_callbacks = []

    def add_risk_event_callback(self, callback: Callable[[str, str], None]):
        """Add callback for risk events."""
        self.on_risk_event_callbacks.append(callback)

    def _trigger_risk_event(self, event_type: str, description: str, severity: str = "warning"):
        """Trigger risk event."""

        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'description': description,
            'severity': severity
        }

        self.risk_events.append(event)

        # Notify callbacks
        for callback in self.on_risk_event_callbacks:
            try:
                callback(event_type, description)
            except Exception as e:
                logger.error(f"Error in risk event callback: {e}")

        # Log event
        metrics_logger.log_risk_event(event_type, description)

        logger.warning(f"Risk Event [{event_type}]: {description}")

    def initialize(self, initial_balance: float):
        """Initialize risk manager with starting balance."""

        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.portfolio_history = [initial_balance]

        logger.info(f"Risk manager initialized with ${initial_balance","} balance")

    def update_portfolio_value(self, current_value: float) -> bool:
        """Update portfolio value and check risk limits."""

        self.portfolio_history.append(current_value)

        # Update peak balance
        if current_value > self.peak_balance:
            self.peak_balance = current_value

        # Check drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_value) / self.peak_balance
            if drawdown > self.max_drawdown:
                self._trigger_risk_event(
                    "max_drawdown_exceeded",
                    f"Portfolio drawdown {drawdown".2%"} exceeds limit {self.max_drawdown".2%"}"
                )
                return True

        # Check daily loss (reset daily if new day)
        now = datetime.now()
        if now.date() != self.last_reset.date():
            self.daily_start_balance = current_value
            self.last_reset = now

        if self.daily_start_balance > 0:
            daily_loss = (self.daily_start_balance - current_value) / self.daily_start_balance
            if daily_loss > self.max_daily_loss:
                self._trigger_risk_event(
                    "max_daily_loss_exceeded",
                    f"Daily loss {daily_loss".2%"} exceeds limit {self.max_daily_loss".2%"}"
                )
                return True

        return False

    def can_open_position(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Check if new position can be opened."""

        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"

        # Check position size limit
        position_value = quantity * price
        portfolio_value = self.portfolio_history[-1] if self.portfolio_history else 0

        if portfolio_value > 0:
            position_size_pct = position_value / portfolio_value
            if position_size_pct > self.max_position_size:
                return False, f"Position size {position_size_pct".2%"} exceeds limit {self.max_position_size".2%"}"

        # Check if symbol already has position
        if symbol in self.positions:
            return False, f"Position already exists for {symbol}"

        return True, "OK"

    def calculate_position_size(self, symbol: str, price: float, volatility: float = 0.02) -> float:
        """Calculate optimal position size based on volatility."""

        portfolio_value = self.portfolio_history[-1] if self.portfolio_history else 0

        if portfolio_value == 0:
            return 0

        # Kelly criterion for position sizing
        # Simplified: position_size = (win_rate * avg_win - loss_rate * avg_loss) / avg_loss
        # For now, use simple volatility-based sizing

        # Base position size
        base_size = portfolio_value * self.max_position_size

        # Adjust for volatility (lower size for higher volatility)
        volatility_adjustment = 1 / (1 + volatility * 10)  # Reduce size as volatility increases

        position_size = base_size * volatility_adjustment

        # Convert to quantity
        quantity = position_size / price

        # Round to reasonable lot size
        quantity = round(quantity, 2)

        return max(0, quantity)

    def should_close_position(self, symbol: str, current_price: float,
                            position: Position) -> Tuple[bool, str]:
        """Check if position should be closed due to risk limits."""

        # Stop loss check
        if position.quantity > 0:  # Long position
            entry_price = position.avg_price
            stop_price = entry_price * (1 - self.stop_loss_pct)

            if current_price <= stop_price:
                return True, f"Stop loss triggered: {current_price".4f"} <= {stop_price".4f"}"

        else:  # Short position
            entry_price = position.avg_price
            stop_price = entry_price * (1 + self.stop_loss_pct)

            if current_price >= stop_price:
                return True, f"Stop loss triggered: {current_price".4f"} >= {stop_price".4f"}"

        # Take profit check
        if position.quantity > 0:  # Long position
            entry_price = position.avg_price
            target_price = entry_price * (1 + self.take_profit_pct)

            if current_price >= target_price:
                return True, f"Take profit triggered: {current_price".4f"} >= {target_price".4f"}"

        else:  # Short position
            entry_price = position.avg_price
            target_price = entry_price * (1 - self.take_profit_pct)

            if current_price <= target_price:
                return True, f"Take profit triggered: {current_price".4f"} <= {target_price".4f"}"

        # Trailing stop check (simplified)
        if self.trailing_stop and position.quantity > 0:
            # Track highest price since entry
            if not hasattr(position, 'highest_price'):
                position.highest_price = position.avg_price

            if current_price > position.highest_price:
                position.highest_price = current_price

            # Trailing stop distance
            trailing_stop_price = position.highest_price * (1 - self.trailing_stop_pct)

            if current_price <= trailing_stop_price:
                return True, f"Trailing stop triggered: {current_price".4f"} <= {trailing_stop_price".4f"}"

        return False, "OK"

    def check_correlation_risk(self, positions: Dict[str, Position]) -> bool:
        """Check for excessive correlation risk in portfolio."""

        if len(positions) < 2:
            return False

        # Simple correlation check - in practice, you'd calculate actual correlations
        # For now, just check if we have too many similar positions

        symbols = list(positions.keys())

        # Check for duplicate symbols (shouldn't happen but good to check)
        if len(symbols) != len(set(symbols)):
            self._trigger_risk_event(
                "duplicate_positions",
                "Duplicate positions detected"
            )
            return True

        return False

    def check_liquidity_risk(self, symbol: str, volume: float) -> bool:
        """Check for liquidity risk."""

        # Simple liquidity check based on volume
        # In practice, you'd check order book depth, spread, etc.

        min_volume_threshold = 1000  # Minimum volume for liquidity

        if volume < min_volume_threshold:
            self._trigger_risk_event(
                "low_liquidity",
                f"Low liquidity for {symbol}: volume {volume} < {min_volume_threshold}"
            )
            return True

        return False

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""

        if not self.portfolio_history:
            return {}

        current_value = self.portfolio_history[-1]

        # Calculate metrics
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        portfolio_metrics = self.metrics.calculate_trade_statistics(returns)

        # Current drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - current_value) / self.peak_balance
        else:
            current_drawdown = 0

        # Daily loss
        if self.daily_start_balance > 0:
            daily_loss = (self.daily_start_balance - current_value) / self.daily_start_balance
        else:
            daily_loss = 0

        return {
            'current_value': current_value,
            'peak_value': self.peak_balance,
            'initial_value': self.initial_balance,
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'num_positions': len(self.positions),
            'portfolio_metrics': portfolio_metrics,
            'risk_events': len(self.risk_events)
        }

    def reset_daily_limits(self):
        """Reset daily risk limits."""

        self.daily_start_balance = self.portfolio_history[-1] if self.portfolio_history else 0
        self.last_reset = datetime.now()

        logger.info("Daily risk limits reset")

    def get_recent_risk_events(self, hours: int = 24) -> List[Dict]:
        """Get risk events from the last N hours."""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_events = [
            event for event in self.risk_events
            if event['timestamp'] >= cutoff_time
        ]

        return recent_events

class PositionSizer:
    """Position sizing strategies."""

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    def size_position_kelly(self, symbol: str, win_rate: float, avg_win: float,
                           avg_loss: float, price: float) -> float:
        """Calculate position size using Kelly criterion."""

        if avg_loss == 0:
            return 0

        # Kelly percentage
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss

        # Cap Kelly at reasonable levels
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Max 25% per position

        # Convert to dollar amount
        portfolio_value = self.risk_manager.portfolio_history[-1] if self.risk_manager.portfolio_history else 0
        position_value = portfolio_value * kelly_pct

        # Convert to quantity
        quantity = position_value / price

        return quantity

    def size_position_volatility(self, symbol: str, price: float, volatility: float) -> float:
        """Calculate position size based on volatility."""

        return self.risk_manager.calculate_position_size(symbol, price, volatility)

    def size_position_fixed(self, symbol: str, price: float, fixed_pct: float = 0.02) -> float:
        """Calculate position size with fixed percentage."""

        portfolio_value = self.risk_manager.portfolio_history[-1] if self.risk_manager.portfolio_history else 0
        position_value = portfolio_value * fixed_pct

        quantity = position_value / price

        return quantity

class RiskMonitor:
    """Monitor and report risk metrics."""

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.monitoring_interval = 60  # seconds

    def start_monitoring(self):
        """Start risk monitoring."""

        # This would run in a separate thread or asyncio task
        # For now, just log current metrics

        metrics = self.risk_manager.get_risk_metrics()

        logger.info("Risk Monitor Update:")
        logger.info(f"  Portfolio Value: ${metrics.get('current_value', 0)",.2f"}")
        logger.info(f"  Current Drawdown: {metrics.get('current_drawdown', 0)".2%"}")
        logger.info(f"  Daily Loss: {metrics.get('daily_loss', 0)".2%"}")
        logger.info(f"  Positions: {metrics.get('num_positions', 0)}")
        logger.info(f"  Risk Events: {metrics.get('risk_events', 0)}")

    def get_alerts(self) -> List[str]:
        """Get current risk alerts."""

        alerts = []
        metrics = self.risk_manager.get_risk_metrics()

        # Drawdown alert
        if metrics.get('current_drawdown', 0) > 0.10:  # 10%
            alerts.append(f"High drawdown: {metrics['current_drawdown']".2%"}")

        # Daily loss alert
        if metrics.get('daily_loss', 0) > 0.03:  # 3%
            alerts.append(f"High daily loss: {metrics['daily_loss']".2%"}")

        # Position concentration alert
        if metrics.get('num_positions', 0) > 10:
            alerts.append(f"High position count: {metrics['num_positions']}")

        return alerts

def create_risk_manager(config: Optional[dict] = None) -> RiskManager:
    """Create risk manager with configuration."""

    return RiskManager(config)
