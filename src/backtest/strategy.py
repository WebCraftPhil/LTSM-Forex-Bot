"""
LSTM trading strategy for backtrader integration.
"""
import backtrader as bt
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from ..utils.config import BacktestConfig
from ..utils.logging import get_logger
from ..models.lstm_fusion import load_model
from ..features.build_dataset import FeatureEngineer

logger = get_logger(__name__)

class LSTMStrategy(bt.Strategy):
    """LSTM-based trading strategy for backtrader."""

    params = (
        ('model_path', None),
        ('config', None),
        ('lookback', 60),  # Lookback period for features
        ('warmup_periods', 100),  # Periods to wait before trading
    )

    def __init__(self):
        super().__init__()

        if self.p.model_path is None:
            raise ValueError("Model path is required")

        # Load LSTM model
        self.model = load_model(self.p.model_path)
        self.model.eval()

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Trading state
        self.warmup_counter = 0
        self.circuit_breaker = False
        self.trades = []

        # Position tracking
        self.current_positions = {}

        # Data storage for feature calculation
        self.price_data = {}
        self.volume_data = {}

        logger.info("LSTM Strategy initialized")

    def next(self):
        """Main strategy logic called for each bar."""

        # Skip warmup period
        if self.warmup_counter < self.p.warmup_periods:
            self.warmup_counter += 1
            return

        # Check circuit breaker
        if self.circuit_breaker:
            return

        # Update market data
        self._update_market_data()

        # Check if we should generate signals
        if self._should_generate_signals():
            # Generate trading signals
            signals = self._generate_signals()

            # Execute trades based on signals
            self._execute_trades(signals)

    def _update_market_data(self):
        """Update price and volume data for all symbols."""

        for data in self.datas:
            symbol = data._name

            if symbol not in self.price_data:
                self.price_data[symbol] = []
                self.volume_data[symbol] = []

            # Store OHLCV data
            self.price_data[symbol].append({
                'open': data.open[0],
                'high': data.high[0],
                'low': data.low[0],
                'close': data.close[0],
                'timestamp': pd.to_datetime(data.datetime.datetime())
            })

            self.volume_data[symbol].append(data.volume[0])

            # Keep only recent data for feature calculation
            max_lookback = self.p.lookback * 4  # 4 timeframes
            if len(self.price_data[symbol]) > max_lookback:
                self.price_data[symbol] = self.price_data[symbol][-max_lookback:]
                self.volume_data[symbol] = self.volume_data[symbol][-max_lookback:]

    def _should_generate_signals(self) -> bool:
        """Check if we should generate trading signals."""

        # Generate signals at the end of each hour (for hourly timeframe)
        current_time = self.data.datetime.time()

        # Simple logic: generate signals every hour
        if current_time.minute == 0:
            return True

        return False

    def _generate_signals(self) -> Dict[str, int]:
        """Generate trading signals using LSTM model."""

        signals = {}

        for symbol in self.price_data.keys():
            try:
                # Create multi-timeframe data
                mtf_data = self._create_mtf_data(symbol)

                if mtf_data is None:
                    continue

                # Build features
                X, _ = self.feature_engineer.build_multi_timeframe_features(
                    {symbol: mtf_data}, sequence_length=self.p.lookback
                )

                if len(X) == 0:
                    continue

                # Use most recent sequence
                latest_X = X[-1:].astype(np.float32)
                latest_X = torch.FloatTensor(latest_X)

                # Get model prediction
                with torch.no_grad():
                    prediction = self.model(latest_X)

                # Convert prediction to signal
                if self.model.config.output_mode == "regression":
                    signal = self._regression_to_signal(prediction.item())
                else:  # classification
                    signal = self._classification_to_signal(prediction)

                signals[symbol] = signal

                logger.info(f"Generated signal for {symbol}: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = 0  # Neutral signal

        return signals

    def _create_mtf_data(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Create multi-timeframe data for feature engineering."""

        if symbol not in self.price_data or len(self.price_data[symbol]) < self.p.lookback:
            return None

        # Convert stored data to DataFrames
        price_df = pd.DataFrame(self.price_data[symbol])
        price_df.set_index('timestamp', inplace=True)

        volume_series = pd.Series(self.volume_data[symbol], index=price_df.index)

        # Create multi-timeframe data (simplified - in practice you'd resample)
        # For this example, we'll use the same timeframe for all
        mtf_data = {
            '1h': price_df.copy(),
        }

        # Add volume to each timeframe
        for tf in mtf_data:
            mtf_data[tf]['volume'] = volume_series

        return mtf_data

    def _regression_to_signal(self, prediction: float) -> int:
        """Convert regression prediction to trading signal."""

        # Simple thresholding
        if prediction > 0.01:  # Strong positive
            return 1  # Long
        elif prediction < -0.01:  # Strong negative
            return -1  # Short
        else:
            return 0  # Neutral

    def _classification_to_signal(self, prediction: torch.Tensor) -> int:
        """Convert classification prediction to trading signal."""

        # Get predicted class
        predicted_class = torch.argmax(prediction, dim=1).item()

        # Convert to signal: 0=neutral, 1=long, 2=short
        if predicted_class == 0:  # Long
            return 1
        elif predicted_class == 1:  # Neutral
            return 0
        elif predicted_class == 2:  # Short
            return -1
        else:
            return 0

    def _execute_trades(self, signals: Dict[str, int]):
        """Execute trades based on signals."""

        for symbol, signal in signals.items():
            current_position = self.getpositionbyname(symbol)

            if signal == 1 and not current_position:  # Long signal and no position
                self._enter_long(symbol)
            elif signal == -1 and not current_position:  # Short signal and no position
                self._enter_short(symbol)
            elif signal == 0 and current_position:  # Neutral signal and position
                self._close_position(symbol)

    def _enter_long(self, symbol: str):
        """Enter long position."""

        # Calculate position size based on risk management
        position_size = self._calculate_position_size(symbol)

        if position_size <= 0:
            return

        # Get current price
        current_price = self.data.close[0]

        # Enter long position
        self.buy(data=self.getdatabyname(symbol), size=position_size)

        # Set stop loss and take profit
        stop_price = current_price * (1 - self.p.config.risk.stop_loss)
        take_profit_price = current_price * (1 + self.p.config.risk.take_profit)

        self.sell(
            data=self.getdatabyname(symbol),
            size=position_size,
            exectype=bt.Order.Stop,
            price=stop_price
        )

        self.sell(
            data=self.getdatabyname(symbol),
            size=position_size,
            exectype=bt.Order.Limit,
            price=take_profit_price
        )

        logger.info(f"Entered long position: {symbol} @ {current_price} (size: {position_size})")

    def _enter_short(self, symbol: str):
        """Enter short position."""

        # Calculate position size
        position_size = self._calculate_position_size(symbol)

        if position_size <= 0:
            return

        # Get current price
        current_price = self.data.close[0]

        # Enter short position
        self.sell(data=self.getdatabyname(symbol), size=position_size)

        # Set stop loss and take profit (for short positions)
        stop_price = current_price * (1 + self.p.config.risk.stop_loss)
        take_profit_price = current_price * (1 - self.p.config.risk.take_profit)

        self.buy(
            data=self.getdatabyname(symbol),
            size=position_size,
            exectype=bt.Order.Stop,
            price=stop_price
        )

        self.buy(
            data=self.getdatabyname(symbol),
            size=position_size,
            exectype=bt.Order.Limit,
            price=take_profit_price
        )

        logger.info(f"Entered short position: {symbol} @ {current_price} (size: {position_size})")

    def _close_position(self, symbol: str):
        """Close existing position."""

        current_position = self.getpositionbyname(symbol)

        if current_position:
            self.close(data=self.getdatabyname(symbol))
            logger.info(f"Closed position: {symbol}")

    def _calculate_position_size(self, symbol: str) -> int:
        """Calculate position size based on risk management."""

        # Get current portfolio value
        portfolio_value = self.broker.getvalue()

        # Calculate maximum position size (2% of portfolio)
        max_position_value = portfolio_value * self.p.config.risk.max_position_size

        # Get current price
        current_price = self.data.close[0]

        # Calculate position size
        position_size = max_position_value / current_price

        # Check position limits
        if len(self.current_positions) >= self.p.config.risk.max_positions:
            return 0

        # Round to reasonable lot size
        position_size = int(position_size / 100) * 100  # Round to hundreds

        return position_size

    def notify_trade(self, trade):
        """Called when a trade is completed."""

        if trade.isclosed:
            # Record trade
            trade_info = {
                'entry_time': pd.to_datetime(trade.dtopen),
                'exit_time': pd.to_datetime(trade.dtclose),
                'pnl': trade.pnlcomm,
                'size': trade.size,
                'entry_price': trade.pricein,
                'exit_price': trade.priceout
            }

            self.trades.append(trade_info)

            logger.info(f"Trade closed: PnL ${trade.pnlcomm".2f"}, Size {trade.size}")

    def notify_order(self, order):
        """Called when an order is completed."""

        if order.status in [order.Completed]:
            logger.info(f"Order completed: {order.ordtypename} {order.size} @ {order.price}")

    def stop(self):
        """Called when backtest ends."""

        logger.info(f"Strategy completed. Final portfolio value: ${self.broker.getvalue()".2f"}")

        # Calculate final statistics
        if self.trades:
            total_pnl = sum(trade['pnl'] for trade in self.trades)
            win_rate = len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)

            logger.info(f"Total trades: {len(self.trades)}")
            logger.info(f"Total P&L: ${total_pnl".2f"}")
            logger.info(f"Win rate: {win_rate".2%"}")
