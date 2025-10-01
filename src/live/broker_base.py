"""
Base broker interface for live trading.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

class Order:
    """Trading order representation."""

    def __init__(self, symbol: str, order_type: str, side: str, quantity: float,
                 price: Optional[float] = None, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None, order_id: Optional[str] = None):
        self.symbol = symbol
        self.order_type = order_type  # market, limit, stop
        self.side = side  # buy, sell
        self.quantity = quantity
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_id = order_id
        self.timestamp = datetime.now()
        self.status = 'pending'

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'symbol': self.symbol,
            'order_type': self.order_type,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'timestamp': self.timestamp,
            'status': self.status
        }

class Position:
    """Trading position representation."""

    def __init__(self, symbol: str, quantity: float, avg_price: float,
                 unrealized_pnl: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.unrealized_pnl = unrealized_pnl
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'timestamp': self.timestamp
        }

class MarketData:
    """Market data representation."""

    def __init__(self, symbol: str, timestamp: datetime, open: float, high: float,
                 low: float, close: float, volume: float, bid: Optional[float] = None,
                 ask: Optional[float] = None):
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.bid = bid
        self.ask = ask

    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }

class BrokerBase(ABC):
    """Abstract base class for broker implementations."""

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        self.api_key = api_key
        self.api_secret = api_secret
        self.connected = False
        self.positions = {}
        self.orders = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker API."""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_orders(self) -> Dict[str, Order]:
        """Get pending orders."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place a trading order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass

    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        pass

    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                                 timeframe: str) -> List[MarketData]:
        """Get historical market data."""
        pass

    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self.connected

    def get_position_value(self) -> float:
        """Get total value of all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_available_cash(self) -> float:
        """Get available cash for trading."""
        # This would be implemented by specific brokers
        return 0.0
