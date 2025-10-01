"""
Real-time market data streaming for live trading.
"""
import asyncio
import websockets
import json
import aiohttp
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
import threading

from ..utils.config import get_config
from ..utils.logging import get_logger
from .broker_base import MarketData

logger = get_logger(__name__)

class MarketDataStreamer:
    """Real-time market data streaming."""

    def __init__(self, broker_adapter, symbols: List[str]):
        self.broker = broker_adapter
        self.symbols = symbols
        self.callbacks = []
        self.running = False
        self.websocket = None
        self.session = None

        # Reconnection settings
        self.reconnect_attempts = 5
        self.reconnect_delay = 5
        self.rate_limit_delay = 0.1

    def add_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for market data updates."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MarketData], None]):
        """Remove callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def start(self):
        """Start streaming market data."""

        if self.running:
            logger.warning("Streamer already running")
            return

        self.running = True

        try:
            # Connect to broker
            if not self.broker.is_connected():
                await self.broker.connect()

            # Start streaming in background
            asyncio.create_task(self._streaming_loop())

            logger.info(f"Started streaming for symbols: {self.symbols}")

        except Exception as e:
            logger.error(f"Error starting streamer: {e}")
            self.running = False
            raise

    async def stop(self):
        """Stop streaming market data."""

        self.running = False

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

        logger.info("Stopped market data streaming")

    async def _streaming_loop(self):
        """Main streaming loop with reconnection logic."""

        attempt = 0

        while self.running and attempt < self.reconnect_attempts:
            try:
                # Connect to websocket
                await self._connect_websocket()

                # Stream data
                await self._stream_data()

            except Exception as e:
                logger.error(f"Streaming error (attempt {attempt + 1}): {e}")
                attempt += 1

                if attempt < self.reconnect_attempts:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Max reconnection attempts reached")
                    break

    async def _connect_websocket(self):
        """Connect to websocket for real-time data."""

        # This would be implemented based on specific broker's websocket API
        # For example, Alpaca uses wss://stream.data.alpaca.markets/v2/{endpoint}

        # For now, simulate websocket connection
        logger.info("Connecting to market data websocket...")

        # In practice:
        # websocket_url = f"wss://stream.data.alpaca.markets/v2/{endpoint}"
        # self.websocket = await websockets.connect(websocket_url)

        await asyncio.sleep(1)  # Simulate connection time

    async def _stream_data(self):
        """Stream market data from websocket."""

        while self.running:
            try:
                # Get latest market data for all symbols
                for symbol in self.symbols:
                    try:
                        market_data = await self.broker.get_market_data(symbol)

                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                callback(market_data)
                            except Exception as e:
                                logger.error(f"Error in callback: {e}")

                    except Exception as e:
                        logger.error(f"Error getting market data for {symbol}: {e}")

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                break

    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to real-time ticker data."""

        # Send subscription message to websocket
        subscription_message = {
            'action': 'subscribe',
            'trades': symbols,
            'quotes': symbols,
            'bars': symbols
        }

        if self.websocket:
            await self.websocket.send(json.dumps(subscription_message))

    async def unsubscribe_ticker(self, symbols: List[str]):
        """Unsubscribe from real-time ticker data."""

        # Send unsubscription message to websocket
        unsubscription_message = {
            'action': 'unsubscribe',
            'trades': symbols,
            'quotes': symbols,
            'bars': symbols
        }

        if self.websocket:
            await self.websocket.send(json.dumps(unsubscription_message))

class DataBuffer:
    """Buffer for storing recent market data."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data = {}

    def add_data(self, market_data: MarketData):
        """Add market data to buffer."""

        symbol = market_data.symbol

        if symbol not in self.data:
            self.data[symbol] = []

        self.data[symbol].append(market_data)

        # Maintain max size
        if len(self.data[symbol]) > self.max_size:
            self.data[symbol] = self.data[symbol][-self.max_size:]

    def get_latest(self, symbol: str, n: int = 1) -> List[MarketData]:
        """Get latest n data points for symbol."""

        if symbol not in self.data:
            return []

        return self.data[symbol][-n:]

    def get_data_range(self, symbol: str, start_time: datetime, end_time: datetime) -> List[MarketData]:
        """Get data within time range."""

        if symbol not in self.data:
            return []

        # Filter by timestamp
        filtered_data = [
            data for data in self.data[symbol]
            if start_time <= data.timestamp <= end_time
        ]

        return filtered_data

class StreamManager:
    """Manager for multiple data streamers."""

    def __init__(self):
        self.streamers = {}
        self.data_buffer = DataBuffer()

    def add_streamer(self, name: str, streamer: MarketDataStreamer):
        """Add a data streamer."""
        self.streamers[name] = streamer

        # Add buffer callback
        streamer.add_callback(self.data_buffer.add_data)

    def get_streamer(self, name: str) -> Optional[MarketDataStreamer]:
        """Get streamer by name."""
        return self.streamers.get(name)

    async def start_all(self):
        """Start all streamers."""
        for name, streamer in self.streamers.items():
            try:
                await streamer.start()
                logger.info(f"Started streamer: {name}")
            except Exception as e:
                logger.error(f"Error starting streamer {name}: {e}")

    async def stop_all(self):
        """Stop all streamers."""
        for name, streamer in self.streamers.items():
            try:
                await streamer.stop()
                logger.info(f"Stopped streamer: {name}")
            except Exception as e:
                logger.error(f"Error stopping streamer {name}: {e}")

    def get_latest_data(self, symbol: str, n: int = 1) -> List[MarketData]:
        """Get latest data for symbol from buffer."""
        return self.data_buffer.get_latest(symbol, n)

    def get_data_range(self, symbol: str, start_time: datetime, end_time: datetime) -> List[MarketData]:
        """Get data range for symbol from buffer."""
        return self.data_buffer.get_data_range(symbol, start_time, end_time)
