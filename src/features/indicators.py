"""
Technical indicators implementation for the LSTM trading bot.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import ta
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """Collection of technical indicators."""

    def rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        try:
            return ta.momentum.RSIIndicator(prices, window=window).rsi()
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(index=prices.index, dtype=float)

    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)."""
        try:
            macd = ta.trend.MACD(prices, window_fast=fast, window_slow=slow, window_sign=signal)
            return pd.DataFrame({
                'macd': macd.macd(),
                'macd_signal': macd.macd_signal(),
                'macd_hist': macd.macd_diff()
            })
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return pd.DataFrame(index=prices.index, columns=['macd', 'macd_signal', 'macd_hist'], dtype=float)

    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator."""
        try:
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=k_window, smooth_window=d_window)
            return pd.DataFrame({
                'stoch_k': stoch.stoch(),
                'stoch_d': stoch.stoch_signal()
            })
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
            return pd.DataFrame(index=close.index, columns=['stoch_k', 'stoch_d'], dtype=float)

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        try:
            return ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return pd.Series(index=close.index, dtype=float)

    def bollinger_bands(self, prices: pd.Series, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
        """Bollinger Bands."""
        try:
            bb = ta.volatility.BollingerBands(prices, window=window, window_dev=window_dev)
            return pd.DataFrame({
                'bb_upper': bb.bollinger_hband(),
                'bb_middle': bb.bollinger_mavg(),
                'bb_lower': bb.bollinger_lband()
            })
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return pd.DataFrame(index=prices.index, columns=['bb_upper', 'bb_middle', 'bb_lower'], dtype=float)

    def sma(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Simple Moving Average."""
        try:
            return ta.trend.SMAIndicator(prices, window=window).sma_indicator()
        except Exception as e:
            logger.warning(f"Error calculating SMA: {e}")
            return pd.Series(index=prices.index, dtype=float)

    def ema(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        try:
            return ta.trend.EMAIndicator(prices, window=window).ema_indicator()
        except Exception as e:
            logger.warning(f"Error calculating EMA: {e}")
            return pd.Series(index=prices.index, dtype=float)

    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        try:
            # Manual VWAP calculation since ta doesn't have it
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return pd.Series(index=close.index, dtype=float)

    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R."""
        try:
            return ta.momentum.WilliamsRIndicator(high, low, close, lbp=window).williams_r()
        except Exception as e:
            logger.warning(f"Error calculating Williams %R: {e}")
            return pd.Series(index=close.index, dtype=float)

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        try:
            return ta.trend.CCIIndicator(high, low, close, window=window).cci()
        except Exception as e:
            logger.warning(f"Error calculating CCI: {e}")
            return pd.Series(index=close.index, dtype=float)

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Movement Index."""
        try:
            return ta.trend.ADXIndicator(high, low, close, window=window).adx()
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return pd.Series(index=close.index, dtype=float)

    def aroon(self, high: pd.Series, low: pd.Series, window: int = 14) -> pd.DataFrame:
        """Aroon Oscillator."""
        try:
            aroon = ta.trend.AroonIndicator(high, low, window=window)
            return pd.DataFrame({
                'aroon_up': aroon.aroon_up(),
                'aroon_down': aroon.aroon_down(),
                'aroon_osc': aroon.aroon_indicator()
            })
        except Exception as e:
            logger.warning(f"Error calculating Aroon: {e}")
            return pd.DataFrame(index=high.index, columns=['aroon_up', 'aroon_down', 'aroon_osc'], dtype=float)

    def ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """Ichimoku Cloud."""
        try:
            # Simplified Ichimoku calculation
            tenkan_window = 9
            kijun_window = 26
            senkou_window = 52

            # Tenkan-sen (Conversion Line)
            tenkan_high = high.rolling(tenkan_window).max()
            tenkan_low = low.rolling(tenkan_window).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2

            # Kijun-sen (Base Line)
            kijun_high = high.rolling(kijun_window).max()
            kijun_low = low.rolling(kijun_window).min()
            kijun_sen = (kijun_high + kijun_low) / 2

            # Senkou Span A (Leading Span A)
            senkou_a = (tenkan_sen + kijun_sen) / 2

            # Senkou Span B (Leading Span B)
            senkou_b = (high.rolling(senkou_window).max() + low.rolling(senkou_window).min()) / 2

            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-26)

            return pd.DataFrame({
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'chikou_span': chikou_span
            })
        except Exception as e:
            logger.warning(f"Error calculating Ichimoku: {e}")
            return pd.DataFrame(index=close.index, columns=['tenkan_sen', 'kijun_sen', 'senkou_a', 'senkou_b', 'chikou_span'], dtype=float)

    def momentum(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Price momentum."""
        try:
            return ta.momentum.ROCIndicator(prices, window=window).roc()
        except Exception as e:
            logger.warning(f"Error calculating momentum: {e}")
            return pd.Series(index=prices.index, dtype=float)

    def volume_sma(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Volume Simple Moving Average."""
        try:
            return ta.volume.VolumeSMAIndicator(volume, window=window).volume_sma()
        except Exception as e:
            logger.warning(f"Error calculating Volume SMA: {e}")
            return pd.Series(index=volume.index, dtype=float)

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume."""
        try:
            return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
            return pd.Series(index=close.index, dtype=float)

    def force_index(self, close: pd.Series, volume: pd.Series, window: int = 2) -> pd.Series:
        """Force Index."""
        try:
            return ta.volume.ForceIndexIndicator(close, volume, window=window).force_index()
        except Exception as e:
            logger.warning(f"Error calculating Force Index: {e}")
            return pd.Series(index=close.index, dtype=float)

    def ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Ease of Movement."""
        try:
            return ta.volume.EaseOfMovementIndicator(high, low, volume, window=window).ease_of_movement()
        except Exception as e:
            logger.warning(f"Error calculating Ease of Movement: {e}")
            return pd.Series(index=high.index, dtype=float)

    def volume_price_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Price Trend."""
        try:
            return ta.volume.VolumePriceTrendIndicator(close, volume).volume_price_trend()
        except Exception as e:
            logger.warning(f"Error calculating Volume Price Trend: {e}")
            return pd.Series(index=close.index, dtype=float)

    def accumulation_distribution(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Index."""
        try:
            return ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
        except Exception as e:
            logger.warning(f"Error calculating Accumulation/Distribution: {e}")
            return pd.Series(index=close.index, dtype=float)

    def chaikin_money_flow(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
        """Chaikin Money Flow."""
        try:
            return ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=window).chaikin_money_flow()
        except Exception as e:
            logger.warning(f"Error calculating Chaikin Money Flow: {e}")
            return pd.Series(index=close.index, dtype=float)

    def volume_rate_of_change(self, volume: pd.Series, window: int = 12) -> pd.Series:
        """Volume Rate of Change."""
        try:
            return ta.volume.VolumeROCIndicator(volume, window=window).volume_roc()
        except Exception as e:
            logger.warning(f"Error calculating Volume ROC: {e}")
            return pd.Series(index=volume.index, dtype=float)

    def price_volume_trend(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Price Volume Trend."""
        try:
            # Manual calculation since ta doesn't have it
            pvt = (close.pct_change() * volume).cumsum()
            return pvt
        except Exception as e:
            logger.warning(f"Error calculating Price Volume Trend: {e}")
            return pd.Series(index=close.index, dtype=float)
