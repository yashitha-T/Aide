"""
Retail-focused, explainable trap feature engineering from OHLCV data.

Features emphasize fake breakouts, volume divergence, liquidity absorption,
and momentum exhaustion using only public OHLCV inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    return np.where(denominator == 0, default, numerator / denominator)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)


def compute_trap_features(
    df: pd.DataFrame,
    breakout_window: int = 20,
    corr_window: int = 10,
    volume_window: int = 20,
    momentum_window: int = 5,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Compute trap-focused features from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        breakout_window: Lookback for breakout highs/lows.
        corr_window: Window for price-volume correlation.
        volume_window: Rolling volume window for spikes/absorption.
        momentum_window: Momentum lookback for exhaustion.
        rsi_period: RSI lookback period.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    features = df.copy()

    features["price_return"] = features["close"].pct_change()
    features["volume_change"] = features["volume"].pct_change()

    rolling_high = features["high"].rolling(window=breakout_window, min_periods=1).max()
    rolling_low = features["low"].rolling(window=breakout_window, min_periods=1).min()

    volume_ma_20 = features["volume"].rolling(window=20, min_periods=1).mean().replace(0, np.nan)

    raw_breakout = _safe_divide(
        features["close"] - rolling_high, rolling_high
    )
    # Required update: breakout strength scaled by (volume / volume_ma_20)
    vol_ratio = (features["volume"] / volume_ma_20).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    features["breakout_strength"] = raw_breakout * vol_ratio

    prev_high = rolling_high.shift(1)
    features["breakout_failure"] = (
        (features["high"] > prev_high) & (features["close"] < prev_high)
    ).astype(int)

    breakout_recent = features["close"].shift(1) > prev_high.shift(1)
    features["failed_retest"] = (
        breakout_recent & (features["close"] < prev_high)
    ).astype(int)

    candle_range = (features["high"] - features["low"]).replace(0, np.nan)
    features["upper_wick_ratio"] = _safe_divide(
        features["high"] - features["close"], candle_range
    )
    features["lower_wick_ratio"] = _safe_divide(
        features["close"] - features["low"], candle_range
    )
    # 3-bar rolling mean — a *sequence* of large wicks is more significant
    features["mean_wick_ratio_3"] = features["upper_wick_ratio"].rolling(3, min_periods=1).mean()

    features["body_to_range_ratio"] = _safe_divide(
        (features["close"] - features["open"]).abs(), candle_range
    )

    volume_mean = features["volume"].rolling(window=volume_window, min_periods=1).mean()
    features["volume_spike"] = (features["volume"] > volume_mean * 1.5).astype(int)

    # Absorption at S/R level — only flag absorption near highs or lows
    near_high = features["close"] >= rolling_high * 0.995
    near_low = features["close"] <= rolling_low * 1.005
    raw_absorption = (
        (features["volume_spike"] == 1) & (features["body_to_range_ratio"] < 0.3)
    )
    features["absorption_signal"] = raw_absorption.astype(int)
    features["absorption_at_level"] = (
        raw_absorption & (near_high | near_low)
    ).astype(int)

    # Required update: continuous volume divergence score.
    # Positive when price rises while volume change weakens.
    pos_return = features["price_return"].clip(lower=0)
    neg_vol_change = (-features["volume_change"]).clip(lower=0)
    max_abs = (
        features["volume_change"]
        .abs()
        .rolling(window=volume_window, min_periods=1)
        .max()
        .replace(0, 1)
    )
    features["volume_divergence"] = (pos_return * (neg_vol_change / max_abs)).clip(0, 1)

    # Fast and slow PV correlation — divergence between them signals regime shift
    features["pv_correlation"] = (
        features["price_return"].rolling(window=corr_window).corr(features["volume_change"])
    )
    features["pv_correlation_fast"] = (
        features["price_return"].rolling(window=5).corr(features["volume_change"])
    )
    features["pv_correlation_slow"] = (
        features["price_return"].rolling(window=20).corr(features["volume_change"])
    )

    features["volume_spike_on_reversal"] = (
        (features["volume_spike"] == 1)
        & (features["price_return"] < 0)
        & (features["price_return"].shift(1) > 0)
    ).astype(int)

    features["gap_rejection"] = (
        (features["open"] > features["close"].shift(1) * 1.002)
        & (features["close"] < features["open"])
    ).astype(int)

    features["price_momentum"] = features["close"].pct_change(periods=momentum_window)
    features["rsi_14"] = _compute_rsi(features["close"], period=rsi_period)
    features["rsi_overbought_fall"] = (
        (features["rsi_14"] > 70) & (features["rsi_14"].diff() < 0)
    ).astype(int)

    # Adaptive fast_reversal: threshold scales with rolling volatility
    # In low-vol environments 1% is significant; in high-vol 3% may be noise
    volatility_20 = features["price_return"].rolling(window=20, min_periods=5).std().fillna(0.01)
    reversal_threshold = (2.5 * volatility_20).clip(lower=0.005, upper=0.05)
    features["fast_reversal"] = (
        (features["price_return"].shift(1) > reversal_threshold)
        & (features["price_return"] < -reversal_threshold)
    ).astype(int)

    features["momentum_fade"] = (
        (features["price_momentum"] < features["price_momentum"].shift(1))
        & (features["price_return"] > 0)
    ).astype(int)

    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    return features
