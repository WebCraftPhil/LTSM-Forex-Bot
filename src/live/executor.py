"""
Live trading executor that integrates models, brokers, and risk management.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta

from ..utils.config import get_config, LiveConfig
from ..utils.logging import get_logger, get_metrics_logger
from ..models.lstm_fusion import load_model
from ..features.build_dataset import FeatureEngineer
from .broker_base import BrokerBase, Order, Position, MarketData
from .streamer import MarketDataStreamer, StreamManager
from ..training.metrics import TradingMetrics

logger = get_logger(__name__)
metrics_logger = get_metrics_logger(__name__)

class CircuitBreaker:
    """Circuit breaker for risk management."""

    def __init__(self, max_drawdown: float = 0.15, max_daily_loss: float = 0.05):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.initial_balance = 0
        self.peak_balance = 0
        self.daily_start_balance = 0
        self.last_reset = datetime.now()
        self.triggered = False

    def update_balance(self, current_balance: float):
        """Update balance and check circuit breaker conditions."""

        # Initialize on first update
        if self.initial_balance == 0:
            self.initial_balance = current_balance
            self.peak_balance = current_balance
            self.daily_start_balance = current_balance
            return False

        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Check drawdown
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            logger.warning(f"Circuit breaker triggered: drawdown {drawdown".2%"} > {self.max_drawdown".2%"}")
            self.triggered = True
            return True

        # Check daily loss (reset daily if new day)
        now = datetime.now()
        if now.date() != self.last_reset.date():
            self.daily_start_balance = current_balance
            self.last_reset = now

        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
        if daily_loss > self.max_daily_loss:
            logger.warning(f"Circuit breaker triggered: daily loss {daily_loss".2%"} > {self.max_daily_loss".2%"}")
            self.triggered = True
            return True

        return False

    def reset(self):
        """Reset circuit breaker."""
        self.triggered = False
        self.peak_balance = self.initial_balance

class TradingExecutor:
    """Main trading executor that coordinates all components."""

    def __init__(self, config: Optional[LiveConfig] = None):
        self.config = config or get_config().live
        self.mode = self.config.mode  # paper, live

        # Initialize components
        self.broker = None
        self.streamer = None
        self.model = None
        self.feature_engineer = FeatureEngineer()

        # Trading state
        self.running = False
        self.positions = {}
        self.orders = {}
        self.last_signal_time = {}

        # Risk management
        self.circuit_breaker = CircuitBreaker(
            max_drawdown=self.config.max_drawdown,
            max_daily_loss=self.config.daily_loss_limit
        )

        # Performance tracking
        self.metrics = TradingMetrics()
        self.start_balance = 0
        self.trades = []

        # Callbacks
        self.on_trade_callbacks = []
        self.on_signal_callbacks = []
        self.on_error_callbacks = []

    async def initialize(self, model_path: str, broker_adapter: BrokerBase):
        """Initialize trading executor."""

        logger.info(f"Initializing trading executor in {self.mode} mode")

        # Load model
        self.model = load_model(model_path)
        logger.info("Model loaded")

        # Setup broker
        self.broker = broker_adapter

        if not await self.broker.connect():
            raise Exception("Failed to connect to broker")

        # Get initial account info
        account_info = await self.broker.get_account_info()
        self.start_balance = account_info.get('portfolio_value', 0)
        self.circuit_breaker.update_balance(self.start_balance)

        logger.info(f"Initial balance: ${self.start_balance","}")

        # Setup streaming
        symbols = self._get_trading_symbols()
        self.streamer = MarketDataStreamer(self.broker, symbols)

        # Add market data callback
        self.streamer.add_callback(self._on_market_data)

        logger.info("Trading executor initialized")

    async def start(self):
        """Start live trading."""

        if self.running:
            logger.warning("Trading executor already running")
            return

        if not self.broker or not self.streamer:
            raise Exception("Executor not initialized")

        self.running = True

        try:
            # Start market data streaming
            await self.streamer.start()

            # Start main trading loop
            asyncio.create_task(self._trading_loop())

            logger.info(f"Started live trading in {self.mode} mode")

        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop live trading."""

        self.running = False

        if self.streamer:
            await self.streamer.stop()

        if self.broker:
            await self.broker.disconnect()

        logger.info("Stopped live trading")

    def add_trade_callback(self, callback: Callable):
        """Add callback for trade events."""
        self.on_trade_callbacks.append(callback)

    def add_signal_callback(self, callback: Callable):
        """Add callback for signal events."""
        self.on_signal_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """Add callback for error events."""
        self.on_error_callbacks.append(callback)

    async def _trading_loop(self):
        """Main trading loop."""

        while self.running:
            try:
                # Check circuit breaker
                if self.circuit_breaker.triggered:
                    logger.warning("Circuit breaker active, skipping trading cycle")
                    await asyncio.sleep(60)  # Wait 1 minute
                    continue

                # Generate trading signals
                await self._generate_signals()

                # Execute pending orders
                await self._execute_orders()

                # Update positions
                await self._update_positions()

                # Check risk limits
                await self._check_risk_limits()

                # Wait before next cycle
                await asyncio.sleep(self.config.execution_delay)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

                # Notify error callbacks
                for callback in self.on_error_callbacks:
                    try:
                        callback(e)
                    except Exception as callback_error:
                        logger.error(f"Error in error callback: {callback_error}")

                await asyncio.sleep(10)  # Wait before retry

    async def _generate_signals(self):
        """Generate trading signals using LSTM model."""

        current_time = datetime.now()

        # Throttle signal generation (e.g., once per minute)
        for symbol in self._get_trading_symbols():
            last_signal = self.last_signal_time.get(symbol, datetime.min)
            if (current_time - last_signal).seconds < 60:
                continue

            try:
                # Get recent market data
                market_data = await self.broker.get_market_data(symbol)

                # Generate signal using model
                signal = await self._generate_model_signal(symbol, market_data)

                if signal != 0:  # Non-neutral signal
                    self.last_signal_time[symbol] = current_time

                    # Notify signal callbacks
                    for callback in self.on_signal_callbacks:
                        try:
                            callback(symbol, signal, market_data.close)
                        except Exception as e:
                            logger.error(f"Error in signal callback: {e}")

                    logger.info(f"Generated signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

    async def _generate_model_signal(self, symbol: str, market_data: MarketData) -> int:
        """Generate signal using LSTM model."""

        # This would integrate with the model to generate signals
        # For now, return a simple signal based on price movement

        # In practice, you would:
        # 1. Get recent price history
        # 2. Build features
        # 3. Run through model
        # 4. Convert prediction to signal

        # Simplified signal generation for demonstration
        if market_data.close > market_data.open:
            return 1  # Long
        elif market_data.close < market_data.open:
            return -1  # Short
        else:
            return 0  # Neutral

    async def _execute_orders(self):
        """Execute pending trading orders."""

        # Get current positions and orders
        self.positions = await self.broker.get_positions()
        self.orders = await self.broker.get_orders()

        # This would contain logic to execute signals as orders
        # For now, it's a placeholder

    async def _update_positions(self):
        """Update position information."""

        # Update circuit breaker with current balance
        account_info = await self.broker.get_account_info()
        current_balance = account_info.get('portfolio_value', 0)

        if self.circuit_breaker.update_balance(current_balance):
            logger.warning("Circuit breaker triggered!")

    async def _check_risk_limits(self):
        """Check and enforce risk limits."""

        # Check position size limits
        for symbol, position in self.positions.items():
            # Implement position size checks
            pass

        # Check daily loss limits
        # Implement daily loss checks
        pass

    def _get_trading_symbols(self) -> List[str]:
        """Get list of symbols to trade."""

        # In practice, this would come from configuration
        return ["EURUSD", "BTCUSD", "ETHUSD"]

    def _on_market_data(self, market_data: MarketData):
        """Handle incoming market data."""

        # Store market data for signal generation
        # This could trigger signal generation if needed
        pass

    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""

        if self.broker:
            # This would get the actual portfolio value from broker
            return self.start_balance

        return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""

        if not self.trades:
            return {}

        trade_returns = [trade['pnl'] / self.start_balance for trade in self.trades]
        return self.metrics.calculate_trade_statistics(np.array(trade_returns))

async def create_paper_trading_executor(model_path: str, api_key: str, api_secret: str) -> TradingExecutor:
    """Create executor for paper trading."""

    from .broker_alpaca import AlpacaBroker

    # Create Alpaca broker in paper trading mode
    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://paper-api.alpaca.markets"
    )

    # Create executor
    executor = TradingExecutor()
    await executor.initialize(model_path, broker)

    return executor

async def create_live_trading_executor(model_path: str, api_key: str, api_secret: str) -> TradingExecutor:
    """Create executor for live trading."""

    from .broker_alpaca import AlpacaBroker

    # Create Alpaca broker in live trading mode
    broker = AlpacaBroker(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://api.alpaca.markets"
    )

    # Create executor
    executor = TradingExecutor()
    await executor.initialize(model_path, broker)

    return executor

def run_executor(model_path: str, mode: str = "paper", api_key: Optional[str] = None, api_secret: Optional[str] = None):
    """Run trading executor (blocking call)."""

    async def main():
        if mode == "paper":
            executor = await create_paper_trading_executor(model_path, api_key, api_secret)
        elif mode == "live":
            executor = await create_live_trading_executor(model_path, api_key, api_secret)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        await executor.start()

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await executor.stop()

    # Run async main function
    asyncio.run(main())
