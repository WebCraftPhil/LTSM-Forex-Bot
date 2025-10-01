"""
Data loaders for historical OHLCV data from various sources.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
import time
import logging

from ..utils.config import get_config, DataConfig
from ..utils.times import ensure_timezone_aware, convert_to_utc, parse_timeframe
from ..utils.logging import get_logger

logger = get_logger(__name__)

class DataLoaderBase:
    """Base class for data loaders."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or get_config().data
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, symbols: List[str], timeframes: List[str],
                  start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load OHLCV data for multiple symbols and timeframes."""
        raise NotImplementedError

    def _get_cache_path(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path."""
        cache_file = f"{symbol}_{timeframe}_{start_date}_{end_date}.parquet"
        return self.cache_dir / cache_file

    def _load_from_cache(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)

        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded {symbol} {timeframe} from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol} {timeframe}: {e}")

        return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str):
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)

        try:
            df.to_parquet(cache_path, compression='snappy')
            logger.info(f"Saved {symbol} {timeframe} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol} {timeframe}: {e}")

class CSVDataLoader(DataLoaderBase):
    """Load OHLCV data from CSV files."""

    def __init__(self, data_dir: str = "data/raw", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)

    def load_data(self, symbols: List[str], timeframes: List[str],
                  start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        result = {}

        for symbol in symbols:
            result[symbol] = {}

            for timeframe in timeframes:
                # Try cache first
                cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                if cached_data is not None:
                    result[symbol][timeframe] = cached_data
                    continue

                # Load from CSV
                df = self._load_symbol_timeframe(symbol, timeframe, start_dt, end_dt)
                if df is not None:
                    result[symbol][timeframe] = df
                    self._save_to_cache(df, symbol, timeframe, start_date, end_date)

        return result

    def _load_symbol_timeframe(self, symbol: str, timeframe: str,
                              start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """Load specific symbol and timeframe from CSV."""

        # Expected filename pattern: SYMBOL_TIMEFRAME.csv
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)

            # Parse timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                logger.warning(f"No timestamp column found in {filepath}")
                return None

            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize('UTC')  # Assume UTC if no timezone

            # Filter date range
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing columns in {filepath}: {missing_cols}")
                return None

            # Sort by timestamp
            df = df.sort_index()

            logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

class AlpacaDataLoader(DataLoaderBase):
    """Load OHLCV data from Alpaca API."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv('API_KEY_ALPACA')
        self.api_secret = api_secret or os.getenv('API_SECRET_ALPACA')

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials required")

    def load_data(self, symbols: List[str], timeframes: List[str],
                  start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:

        # Alpaca uses different timeframe notation
        alpaca_timeframes = {
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1H',
            '2h': '2H'
        }

        result = {}

        for symbol in symbols:
            result[symbol] = {}

            for timeframe in timeframes:
                if timeframe not in alpaca_timeframes:
                    logger.warning(f"Unsupported timeframe for Alpaca: {timeframe}")
                    continue

                alpaca_tf = alpaca_timeframes[timeframe]

                # Try cache first
                cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                if cached_data is not None:
                    result[symbol][timeframe] = cached_data
                    continue

                # Load from Alpaca API
                df = self._load_alpaca_data(symbol, alpaca_tf, start_date, end_date)
                if df is not None:
                    result[symbol][timeframe] = df
                    self._save_to_cache(df, symbol, timeframe, start_date, end_date)

        return result

    def _load_alpaca_data(self, symbol: str, timeframe: str,
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from Alpaca API."""

        # Note: This is a simplified implementation
        # In production, you'd use the official alpaca-py library

        base_url = "https://data.alpaca.markets/v1"

        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }

        # Alpaca bars endpoint
        url = f"{base_url}/bars/{timeframe}"

        params = {
            'symbols': symbol,
            'start': start_date,
            'end': end_date,
            'limit': 10000  # Alpaca limit
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            if symbol not in data or not data[symbol]:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            bars = data[symbol]
            df = pd.DataFrame(bars)

            # Rename columns to standard format
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Ensure timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York').tz_convert('UTC')

            logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading Alpaca data for {symbol}: {e}")
            return None

class BinanceDataLoader(DataLoaderBase):
    """Load OHLCV data from Binance API."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv('API_KEY_BINANCE')
        self.api_secret = api_secret or os.getenv('API_SECRET_BINANCE')

        # Binance uses different symbol format (BTCUSDT vs BTCUSD)
        self.symbol_map = {
            'BTCUSD': 'BTCUSDT',
            'ETHUSD': 'ETHUSDT',
            'EURUSD': 'EURUSDT',  # May not exist
        }

    def load_data(self, symbols: List[str], timeframes: List[str],
                  start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:

        # Binance timeframe mapping
        binance_timeframes = {
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h'
        }

        result = {}

        for symbol in symbols:
            binance_symbol = self.symbol_map.get(symbol, symbol)
            result[symbol] = {}

            for timeframe in timeframes:
                if timeframe not in binance_timeframes:
                    logger.warning(f"Unsupported timeframe for Binance: {timeframe}")
                    continue

                # Try cache first
                cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                if cached_data is not None:
                    result[symbol][timeframe] = cached_data
                    continue

                # Load from Binance API
                df = self._load_binance_data(binance_symbol, timeframe, start_date, end_date)
                if df is not None:
                    result[symbol][timeframe] = df
                    self._save_to_cache(df, symbol, timeframe, start_date, end_date)

        return result

    def _load_binance_data(self, symbol: str, timeframe: str,
                          start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from Binance API."""

        base_url = "https://api.binance.com/api/v3/klines"

        # Convert dates to timestamps (milliseconds)
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        params = {
            'symbol': symbol,
            'interval': timeframe,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000  # Binance limit per request
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            df[price_cols] = df[price_cols].astype(float)

            # Binance returns UTC timestamps
            df.index = df.index.tz_localize('UTC')

            logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading Binance data for {symbol}: {e}")
            return None

class OANDADataLoader(DataLoaderBase):
    """Load OHLCV data from OANDA API."""

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv('API_KEY_OANDA')
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA API credentials required")

    def load_data(self, symbols: List[str], timeframes: List[str],
                  start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:

        # OANDA timeframe mapping
        oanda_timeframes = {
            '15m': 'M15',
            '30m': 'M30',
            '1h': 'H1',
            '2h': 'H2'
        }

        result = {}

        for symbol in symbols:
            result[symbol] = {}

            for timeframe in timeframes:
                if timeframe not in oanda_timeframes:
                    logger.warning(f"Unsupported timeframe for OANDA: {timeframe}")
                    continue

                oanda_tf = oanda_timeframes[timeframe]

                # Try cache first
                cached_data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                if cached_data is not None:
                    result[symbol][timeframe] = cached_data
                    continue

                # Load from OANDA API
                df = self._load_oanda_data(symbol, oanda_tf, start_date, end_date)
                if df is not None:
                    result[symbol][timeframe] = df
                    self._save_to_cache(df, symbol, timeframe, start_date, end_date)

        return result

    def _load_oanda_data(self, symbol: str, timeframe: str,
                        start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from OANDA API."""

        base_url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        params = {
            'granularity': timeframe,
            'from': start_date,
            'to': end_date,
            'price': 'M',  # Midpoint candles
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            if 'candles' not in data or not data['candles']:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            candles = []
            for candle in data['candles']:
                if candle['complete']:
                    candles.append({
                        'timestamp': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': float(candle['volume'])
                    })

            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error loading OANDA data for {symbol}: {e}")
            return None

def get_data_loader(source: str = None, **kwargs) -> DataLoaderBase:
    """Factory function to get appropriate data loader."""

    if source is None:
        source = get_config().data.data_source

    loaders = {
        'csv': CSVDataLoader,
        'alpaca': AlpacaDataLoader,
        'binance': BinanceDataLoader,
        'oanda': OANDADataLoader
    }

    if source not in loaders:
        raise ValueError(f"Unsupported data source: {source}")

    return loaders[source](**kwargs)

def load_ohlcv_data(symbols: List[str], timeframes: List[str],
                   start_date: str, end_date: str,
                   source: str = None, **kwargs) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load OHLCV data from specified source."""

    loader = get_data_loader(source, **kwargs)
    return loader.load_data(symbols, timeframes, start_date, end_date)
