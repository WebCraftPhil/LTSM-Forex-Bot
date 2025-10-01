"""
Alpaca broker implementation for live trading.
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import os

from .broker_base import BrokerBase, Order, Position, MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)

class AlpacaBroker(BrokerBase):
    """Alpaca broker implementation."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 base_url: str = "https://paper-api.alpaca.markets", **kwargs):
        super().__init__(api_key or os.getenv('API_KEY_ALPACA'),
                        api_secret or os.getenv('API_SECRET_ALPACA'), **kwargs)

        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

        # API endpoints
        self.account_url = f"{self.base_url}/v2/account"
        self.orders_url = f"{self.base_url}/v2/orders"
        self.positions_url = f"{self.base_url}/v2/positions"
        self.market_data_url = f"{self.base_url}/v2/stocks"

    async def connect(self) -> bool:
        """Connect to Alpaca API."""

        try:
            self.session = aiohttp.ClientSession()

            # Test connection with account info
            account = await self.get_account_info()

            if account:
                self.connected = True
                logger.info("Connected to Alpaca API")
                return True
            else:
                logger.error("Failed to connect to Alpaca API")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Alpaca API."""

        try:
            if self.session:
                await self.session.close()
            self.connected = False
            logger.info("Disconnected from Alpaca API")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {e}")
            return False

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""

        if not self.connected or not self.session:
            return {}

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        try:
            async with self.session.get(self.account_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'account_id': data.get('id'),
                        'cash': float(data.get('cash', 0)),
                        'portfolio_value': float(data.get('portfolio_value', 0)),
                        'buying_power': float(data.get('buying_power', 0)),
                        'currency': data.get('currency', 'USD')
                    }
                else:
                    logger.error(f"Error getting account info: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""

        if not self.connected or not self.session:
            return {}

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        try:
            async with self.session.get(self.positions_url, headers=headers) as response:
                if response.status == 200:
                    positions_data = await response.json()

                    positions = {}
                    for pos_data in positions_data:
                        symbol = pos_data['symbol']
                        positions[symbol] = Position(
                            symbol=symbol,
                            quantity=float(pos_data['qty']),
                            avg_price=float(pos_data['avg_entry_price']),
                            unrealized_pnl=float(pos_data['unrealized_pl'])
                        )

                    self.positions = positions
                    return positions
                else:
                    logger.error(f"Error getting positions: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    async def get_orders(self) -> Dict[str, Order]:
        """Get pending orders."""

        if not self.connected or not self.session:
            return {}

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        params = {'status': 'open'}

        try:
            async with self.session.get(self.orders_url, headers=headers, params=params) as response:
                if response.status == 200:
                    orders_data = await response.json()

                    orders = {}
                    for order_data in orders_data:
                        order_id = order_data['id']
                        orders[order_id] = Order(
                            symbol=order_data['symbol'],
                            order_type=order_data['type'],
                            side=order_data['side'],
                            quantity=float(order_data['qty']),
                            price=float(order_data['limit_price']) if order_data.get('limit_price') else None,
                            order_id=order_id,
                            status=order_data['status']
                        )

                    self.orders = orders
                    return orders
                else:
                    logger.error(f"Error getting orders: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return {}

    async def place_order(self, order: Order) -> str:
        """Place a trading order."""

        if not self.connected or not self.session:
            raise Exception("Not connected to broker")

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }

        # Build order payload
        payload = {
            'symbol': order.symbol,
            'qty': order.quantity,
            'side': order.side,
            'type': order.order_type,
            'time_in_force': 'day'
        }

        if order.price:
            payload['limit_price'] = order.price

        if order.stop_loss:
            payload['stop_loss'] = {'stop_price': order.stop_loss}

        if order.take_profit:
            payload['take_profit'] = {'limit_price': order.take_profit}

        try:
            async with self.session.post(self.orders_url, headers=headers, json=payload) as response:
                if response.status in [201, 200]:
                    order_data = await response.json()
                    order_id = order_data['id']
                    order.order_id = order_id
                    order.status = 'pending'

                    logger.info(f"Placed order: {order_id} for {order.symbol}")
                    return order_id
                else:
                    error_text = await response.text()
                    logger.error(f"Error placing order: {response.status} - {error_text}")
                    raise Exception(f"Failed to place order: {error_text}")

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

        if not self.connected or not self.session:
            return False

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        try:
            async with self.session.delete(f"{self.orders_url}/{order_id}", headers=headers) as response:
                if response.status == 204:
                    logger.info(f"Cancelled order: {order_id}")
                    return True
                else:
                    logger.error(f"Error cancelling order {order_id}: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""

        if not self.connected or not self.session:
            raise Exception("Not connected to broker")

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        try:
            async with self.session.get(f"{self.market_data_url}/{symbol}/quotes/latest", headers=headers) as response:
                if response.status == 200:
                    quote_data = await response.json()

                    quote = quote_data['quote']
                    return MarketData(
                        symbol=symbol,
                        timestamp=pd.to_datetime(quote['t']),
                        bid=quote['bp'],
                        ask=quote['ap'],
                        # Note: Alpaca doesn't provide OHLCV in quotes, would need separate call
                        open=0, high=0, low=0, close=0, volume=0
                    )
                else:
                    logger.error(f"Error getting market data for {symbol}: {response.status}")
                    raise Exception(f"Failed to get market data for {symbol}")

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise

    async def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                                 timeframe: str) -> List[MarketData]:
        """Get historical market data."""

        if not self.connected or not self.session:
            raise Exception("Not connected to broker")

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # Convert timeframe to Alpaca format
        timeframe_map = {
            '1m': '1Min', '5m': '5Min', '15m': '15Min', '30m': '30Min',
            '1h': '1H', '1d': '1D', '1w': '1W', '1M': '1M'
        }

        alpaca_timeframe = timeframe_map.get(timeframe, timeframe)

        params = {
            'start': start_date,
            'end': end_date,
            'timeframe': alpaca_timeframe
        }

        try:
            # Note: This would use Alpaca's historical data API
            # For now, return empty list as implementation would depend on specific API
            logger.warning("Historical data not implemented for Alpaca")
            return []

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
