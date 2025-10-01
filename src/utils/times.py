"""
Time utilities for the LSTM trading bot.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import pytz
from datetime import datetime, timedelta

# Default timezone for trading data
DEFAULT_TZ = pytz.UTC

def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone aware."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt

def convert_to_utc(dt: datetime, from_tz: str = 'UTC') -> datetime:
    """Convert datetime to UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.timezone(from_tz))

    return dt.astimezone(pytz.UTC)

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """Parse timeframe string into minutes and unit.

    Args:
        timeframe: Timeframe string like '15m', '1h', '2h', '1d'

    Returns:
        Tuple of (minutes, unit)
    """
    unit_multipliers = {
        'm': 1,
        'h': 60,
        'd': 1440,  # 24 * 60
        'w': 10080,  # 7 * 24 * 60
    }

    if timeframe[-1] not in unit_multipliers:
        raise ValueError(f"Invalid timeframe unit: {timeframe}")

    value = int(timeframe[:-1])
    unit = timeframe[-1]

    return value * unit_multipliers[unit], unit

def get_timeframe_minutes(timeframe: str) -> int:
    """Get timeframe in minutes."""
    minutes, _ = parse_timeframe(timeframe)
    return minutes

def align_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """Align datetime to timeframe boundary."""
    minutes, unit = parse_timeframe(timeframe)

    if unit == 'm':
        # Align to minute boundary
        aligned_minute = (dt.minute // minutes) * minutes
        return dt.replace(minute=aligned_minute, second=0, microsecond=0)

    elif unit == 'h':
        # Align to hour boundary
        aligned_hour = (dt.hour // minutes) * minutes
        return dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)

    elif unit == 'd':
        # Align to day boundary (assuming minutes represents hours in day)
        aligned_hour = (dt.hour // minutes) * minutes
        return dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)

    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def resample_timeframe(df: pd.DataFrame, from_tf: str, to_tf: str) -> pd.DataFrame:
    """Resample dataframe from one timeframe to another.

    Args:
        df: DataFrame with datetime index
        from_tf: Source timeframe
        to_tf: Target timeframe

    Returns:
        Resampled DataFrame
    """
    from_minutes = get_timeframe_minutes(from_tf)
    to_minutes = get_timeframe_minutes(to_tf)

    if to_minutes < from_minutes:
        raise ValueError(f"Cannot downsample from {from_tf} to {to_tf}")

    # Calculate resampling factor
    factor = to_minutes // from_minutes

    # Resample by taking every Nth row where N = factor
    resampled = df.iloc[::factor].copy()

    # Update index to reflect new timeframe
    resampled.index = resampled.index.map(
        lambda x: align_to_timeframe(x, to_tf)
    )

    return resampled

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime index.

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with additional time features
    """
    df = df.copy()

    # Basic time features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # Trading session features
    df['is_market_open'] = df.index.hour.between(0, 23)  # 24/7 for crypto/forex

    # Weekend indicator
    df['is_weekend'] = df.index.dayofweek.isin([5, 6])

    # Session indicators (simplified)
    df['is_asian_session'] = df.index.hour.between(0, 8)
    df['is_european_session'] = df.index.hour.between(8, 16)
    df['is_us_session'] = df.index.hour.between(16, 23)

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    return df

def create_rolling_windows(df: pd.DataFrame, windows: List[int],
                          features: Optional[List[str]] = None) -> pd.DataFrame:
    """Create rolling window features.

    Args:
        df: DataFrame with OHLCV data
        windows: List of window sizes in periods
        features: List of features to create rolling stats for

    Returns:
        DataFrame with rolling features
    """
    if features is None:
        features = ['close', 'volume', 'high', 'low']

    df = df.copy()
    rolling_features = []

    for window in windows:
        for feature in features:
            if feature in df.columns:
                # Rolling mean
                df[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window).mean()

                # Rolling std
                df[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window).std()

                # Rolling min/max
                df[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window=window).min()
                df[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window=window).max()

                # Rolling volatility (std/mean)
                df[f'{feature}_rolling_vol_{window}'] = (
                    df[f'{feature}_rolling_std_{window}'] / df[f'{feature}_rolling_mean_{window}']
                )

                rolling_features.extend([
                    f'{feature}_rolling_mean_{window}',
                    f'{feature}_rolling_std_{window}',
                    f'{feature}_rolling_min_{window}',
                    f'{feature}_rolling_max_{window}',
                    f'{feature}_rolling_vol_{window}'
                ])

    return df

def shift_features(df: pd.DataFrame, periods: List[int],
                  features: Optional[List[str]] = None) -> pd.DataFrame:
    """Create lagged features by shifting columns.

    Args:
        df: DataFrame with features
        periods: List of periods to shift by
        features: List of features to shift (if None, shift all numeric columns)

    Returns:
        DataFrame with lagged features
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.copy()
    lagged_features = []

    for period in periods:
        for feature in features:
            if feature in df.columns:
                df[f'{feature}_lag_{period}'] = df[feature].shift(period)
                lagged_features.append(f'{feature}_lag_{period}')

    return df
