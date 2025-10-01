"""
Feature engineering for multi-timeframe LSTM trading bot.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import ta
import logging

from ..utils.config import get_config
from ..utils.times import create_time_features, create_rolling_windows, shift_features
from ..utils.logging import get_logger
from .indicators import TechnicalIndicators

logger = get_logger(__name__)

class FeatureEngineer:
    """Build comprehensive features for multi-timeframe trading."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.indicators = TechnicalIndicators()

    def build_multi_timeframe_features(self, data: Dict[str, Dict[str, pd.DataFrame]],
                                     sequence_length: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
        """Build features and targets from multi-timeframe data.

        Args:
            data: Nested dict with {symbol: {timeframe: DataFrame}}
            sequence_length: Length of sequences for LSTM

        Returns:
            Tuple of (features DataFrame, targets Series)
        """

        # Extract first symbol and its timeframes for processing
        # In production, you'd process all symbols
        symbol = list(data.keys())[0]
        timeframes_data = data[symbol]

        logger.info(f"Building features for {symbol} with timeframes: {list(timeframes_data.keys())}")

        # Process each timeframe
        tf_features = {}
        for tf, df in timeframes_data.items():
            tf_features[tf] = self._build_single_timeframe_features(df, tf)

        # Align timeframes and create multi-TF features
        aligned_features = self._align_timeframes(tf_features)

        # Create sequences for LSTM
        X, y = self._create_sequences(aligned_features, sequence_length)

        logger.info(f"Created {len(X)} sequences with {X.shape[1] if len(X) > 0 else 0} features each")
        return X, y

    def _build_single_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Build features for a single timeframe."""

        df = df.copy()

        # Basic OHLCV features
        features_df = self._add_ohlcv_features(df)

        # Technical indicators
        features_df = self._add_technical_indicators(features_df)

        # Time features
        features_df = self._add_time_features(features_df)

        # Rolling statistics
        features_df = self._add_rolling_features(features_df)

        # Lagged features
        features_df = self._add_lagged_features(features_df)

        # Price ratios and returns
        features_df = self._add_return_features(features_df)

        # Volume features
        features_df = self._add_volume_features(features_df)

        # Volatility features
        features_df = self._add_volatility_features(features_df)

        # Clean up features
        features_df = self._clean_features(features_df)

        logger.info(f"Built {features_df.shape[1]} features for {timeframe}")
        return features_df

    def _add_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic OHLCV-based features."""
        df = df.copy()

        # Price differences
        df['price_change'] = df['close'] - df['open']
        df['high_low_diff'] = df['high'] - df['low']
        df['close_open_ratio'] = df['close'] / df['open'] - 1

        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Price position within day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA library."""

        indicators_config = self.config.data.features.get('technical_indicators', [])

        if not indicators_config:
            return df

        df = df.copy()

        # RSI
        if 'rsi' in indicators_config:
            df['rsi'] = self.indicators.rsi(df['close'])

        # MACD
        if 'macd' in indicators_config:
            macd_df = self.indicators.macd(df['close'])
            df['macd'] = macd_df['macd']
            df['macd_signal'] = macd_df['macd_signal']
            df['macd_hist'] = macd_df['macd_hist']

        # Stochastic
        if 'stochastic' in indicators_config:
            stoch_df = self.indicators.stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_df['stoch_k']
            df['stoch_d'] = stoch_df['stoch_d']

        # ATR
        if 'atr' in indicators_config:
            df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'])

        # Bollinger Bands
        if 'bollinger_bands' in indicators_config:
            bb_df = self.indicators.bollinger_bands(df['close'])
            df['bb_upper'] = bb_df['bb_upper']
            df['bb_middle'] = bb_df['bb_middle']
            df['bb_lower'] = bb_df['bb_lower']
            df['bb_position'] = (df['close'] - bb_df['bb_middle']) / (bb_df['bb_upper'] - bb_df['bb_lower'])

        # Moving Averages
        if 'sma_fast' in indicators_config:
            df['sma_fast'] = self.indicators.sma(df['close'], window=10)

        if 'sma_slow' in indicators_config:
            df['sma_slow'] = self.indicators.sma(df['close'], window=50)

        if 'ema_fast' in indicators_config:
            df['ema_fast'] = self.indicators.ema(df['close'], window=10)

        if 'ema_slow' in indicators_config:
            df['ema_slow'] = self.indicators.ema(df['close'], window=50)

        # VWAP (Volume Weighted Average Price)
        if 'vwap' in indicators_config:
            df['vwap'] = self.indicators.vwap(df['high'], df['low'], df['close'], df['volume'])

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        return create_time_features(df)

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        windows = [5, 10, 20, 50]
        return create_rolling_windows(df, windows)

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        lag_config = self.config.data.features.get('lagged_features', {})

        periods = []
        if 'returns' in lag_config:
            periods.extend(lag_config['returns'])
        if 'volume' in lag_config:
            periods.extend(lag_config['volume'])

        if periods:
            return shift_features(df, periods)

        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        df = df.copy()

        # Returns at different horizons
        for period in [1, 5, 10, 20, 50]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # Cumulative returns
        df['return_cum_5'] = df['close'].pct_change(5).rolling(5).sum()
        df['return_cum_10'] = df['close'].pct_change(10).rolling(10).sum()

        # Drawdown features
        rolling_max = df['close'].rolling(50, min_periods=1).max()
        df['drawdown'] = (df['close'] - rolling_max) / rolling_max

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Volume price trend
        df['volume_price_trend'] = (df['volume'] * df['close'].pct_change()).rolling(10).sum()

        # Volume spikes
        volume_mean = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        df['volume_spike'] = (df['volume'] - volume_mean) / volume_std

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        df = df.copy()

        # Rolling volatility
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # Parkinson volatility (high-low based)
        df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(df['high']/df['low']))**2).rolling(20).mean())

        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            (0.5 * (np.log(df['high']/df['low']))**2) -
            (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
        ).rolling(20).mean()

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features."""

        # Remove original OHLCV columns (keep derived features only)
        keep_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        # Drop rows with NaN values (from rolling windows, indicators)
        df_clean = df[keep_columns].dropna()

        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(f"Cleaned features: {df_clean.shape[1]} columns, {len(df_clean)} rows")
        return df_clean

    def _align_timeframes(self, tf_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align features from different timeframes to common timestamp index."""

        # Find the finest timeframe (smallest interval)
        timeframes = list(tf_features.keys())
        finest_tf = min(timeframes, key=lambda x: int(x[:-1]) * (60 if x.endswith('m') else 1))

        # Use finest timeframe as base
        base_df = tf_features[finest_tf]

        # Align other timeframes to base timeframe
        aligned_features = [base_df]

        for tf in timeframes:
            if tf != finest_tf:
                # Resample to finest timeframe (forward fill)
                aligned = tf_features[tf].reindex(base_df.index, method='ffill')
                aligned_features.append(aligned)

        # Combine all features
        result = pd.concat(aligned_features, axis=1)

        # Add timeframe suffix to avoid column name conflicts
        for i, tf in enumerate(timeframes):
            if tf != finest_tf:
                tf_cols = [col for col in result.columns if col in tf_features[tf].columns]
                result.rename(columns={col: f"{col}_{tf}" for col in tf_cols}, inplace=True)

        return result

    def _create_sequences(self, features: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.Series]:
        """Create sequences for LSTM training."""

        # Create target (next period return)
        target = features['close'].pct_change().shift(-1)

        # Remove target column from features
        feature_cols = [col for col in features.columns if col != 'close']
        X_data = features[feature_cols].values

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(len(X_data) - sequence_length):
            X_seq = X_data[i:(i + sequence_length)]
            y_seq = target.iloc[i + sequence_length]

            # Skip if target is NaN
            if not pd.isna(y_seq):
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        X = np.array(X_sequences)
        y = pd.Series(y_sequences)

        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y

    def create_target_labels(self, returns: pd.Series, n_classes: int = 3) -> pd.Series:
        """Create classification labels from returns."""

        if n_classes == 3:
            # Long/Flat/Short classification
            labels = pd.cut(returns, bins=[-np.inf, -0.001, 0.001, np.inf], labels=[-1, 0, 1])
        else:
            # Multi-class based on quantiles
            quantiles = np.linspace(0, 1, n_classes + 1)
            bins = returns.quantile(quantiles).values
            labels = pd.cut(returns, bins=bins, labels=range(n_classes))

        return labels.astype(int)

def build_dataset(data: Dict[str, Dict[str, pd.DataFrame]], sequence_length: int = 60) -> Tuple[np.ndarray, pd.Series]:
    """Build complete dataset from multi-timeframe data."""

    engineer = FeatureEngineer()
    X, y = engineer.build_multi_timeframe_features(data, sequence_length)

    return X, y

def save_dataset(X: np.ndarray, y: pd.Series, filepath: str):
    """Save dataset to disk."""

    import pickle

    dataset = {
        'X': X,
        'y': y,
        'timestamp': pd.Timestamp.now()
    }

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(filepath: str) -> Tuple[np.ndarray, pd.Series]:
    """Load dataset from disk."""

    import pickle

    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)

    return dataset['X'], dataset['y']
