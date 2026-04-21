"""Retail behavior modeling features for MarketTrap."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _sigmoid(x: float | np.ndarray, center: float = 0.0, steepness: float = 5.0):
    z = steepness * (x - center)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_taker_ratio(
    df: pd.DataFrame,
    taker_buy_col: str = "taker_buy_volume",
    taker_sell_col: str = "taker_sell_volume",
    volume_col: str = "volume",
) -> pd.DataFrame:
    out = df.copy()
    vol = out[volume_col].astype(float).replace(0, np.nan)

    if taker_buy_col in out.columns:
        out["taker_buy_ratio"] = (out[taker_buy_col].astype(float) / vol).clip(0, 1).fillna(0.5)
    else:
        out["taker_buy_ratio"] = 0.5

    if taker_sell_col in out.columns:
        out["taker_sell_ratio"] = (out[taker_sell_col].astype(float) / vol).clip(0, 1).fillna(0.5)
    else:
        out["taker_sell_ratio"] = (1.0 - out["taker_buy_ratio"]).clip(0, 1)

    return out


def compute_fomo_index(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    fast_window: int = 5,
    slow_window: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    prices = out[price_col].astype(float)
    volumes = out[volume_col].astype(float)

    vol_ma_fast = volumes.rolling(window=fast_window, min_periods=1).mean()
    vol_ma_slow = volumes.rolling(window=slow_window, min_periods=1).mean().replace(0, np.nan)
    out["volume_acceleration"] = ((vol_ma_fast / vol_ma_slow) - 1).fillna(0)

    ret_fast = prices.pct_change(periods=fast_window).fillna(0)
    ret_slow = prices.pct_change(periods=slow_window).fillna(0)
    out["price_acceleration"] = np.where(
        ret_slow.abs() > 0.0001,
        (ret_fast / ret_slow.replace(0, np.nan)).fillna(0),
        0.0,
    )

    if "taker_buy_ratio" not in out.columns:
        out = compute_taker_ratio(out, volume_col=volume_col)

    fomo = (
        0.40 * _sigmoid(out["volume_acceleration"].values, center=0.5, steepness=4)
        + 0.30 * _sigmoid(out["price_acceleration"].values, center=0.3, steepness=5)
        + 0.30 * _sigmoid(out["taker_buy_ratio"].values - 0.55, center=0.0, steepness=10)
    )
    out["fomo_index"] = _clip01(fomo)

    rolling_high = prices.rolling(window=slow_window, min_periods=5).max()
    near_resistance = prices >= rolling_high * 0.995
    out["fomo_at_resistance"] = (out["fomo_index"] * near_resistance.astype(float)).clip(0, 1)
    return out


def compute_panic_index(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "volume",
    volatility_window: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    prices = out[price_col].astype(float)
    volumes = out[volume_col].astype(float)

    returns = prices.pct_change().fillna(0)
    vol_std = returns.rolling(window=volatility_window, min_periods=5).std().fillna(0.01)
    vol_ma = volumes.rolling(window=volatility_window, min_periods=1).mean().replace(0, 1)

    sharp_drop = (returns < -2 * vol_std).astype(float)
    vol_ratio = volumes / vol_ma

    if "taker_sell_ratio" not in out.columns:
        out = compute_taker_ratio(out, volume_col=volume_col)

    panic = (
        0.35 * sharp_drop.values
        + 0.35 * _sigmoid(vol_ratio.values, center=2.0, steepness=3)
        + 0.30 * _sigmoid(out["taker_sell_ratio"].values - 0.55, center=0.0, steepness=10)
    )
    out["panic_index"] = _clip01(panic)

    rolling_low = prices.rolling(window=volatility_window, min_periods=5).min()
    near_support = prices <= rolling_low * 1.005
    out["panic_at_support"] = (out["panic_index"] * near_support.astype(float)).clip(0, 1)
    return out


def compute_crowd_positioning(df: pd.DataFrame, derivatives_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    derivatives_data format:
    {
      "ls_series": [...],
      "funding_series": [...],
      "oi_series": [...]
    }
    """
    out = df.copy()
    derivatives_data = derivatives_data or {}

    ls_series: List[Dict] = derivatives_data.get("ls_series", []) or []
    funding_series: List[Dict] = derivatives_data.get("funding_series", []) or []
    oi_series: List[Dict] = derivatives_data.get("oi_series", []) or []

    ls_val = float(ls_series[-1].get("long_short_ratio", 1.0)) if ls_series else 1.0
    fr_val = float(funding_series[-1].get("funding_rate", 0.0)) if funding_series else 0.0

    oi_bias = 0.0
    if len(oi_series) >= 5:
        oi_start = float(oi_series[-5].get("open_interest", 0.0))
        oi_end = float(oi_series[-1].get("open_interest", 0.0))
        if oi_start != 0:
            oi_bias = np.clip((oi_end - oi_start) / abs(oi_start), -0.2, 0.2) / 0.2

    ls_zscore = (ls_val - 1.0) / 0.3
    fr_zscore = (fr_val - 0.0001) / max(0.0003, abs(fr_val) * 0.5 + 0.0001)

    raw = (
        0.40 * np.clip(ls_zscore, -3, 3) / 3.0
        + 0.40 * np.clip(fr_zscore, -3, 3) / 3.0
        + 0.20 * oi_bias
    )

    out["crowd_positioning_score"] = float(np.clip(raw, -1, 1))
    return out


def compute_all_retail_features(
    df: pd.DataFrame,
    ls_series: Optional[List[Dict]] = None,
    funding_series: Optional[List[Dict]] = None,
    oi_series: Optional[List[Dict]] = None,
    price_col: str = "price",
    volume_col: str = "volume",
) -> pd.DataFrame:
    out = compute_taker_ratio(df, volume_col=volume_col)
    out = compute_fomo_index(out, price_col=price_col, volume_col=volume_col)
    out = compute_panic_index(out, price_col=price_col, volume_col=volume_col)
    out = compute_crowd_positioning(
        out,
        derivatives_data={
            "ls_series": ls_series or [],
            "funding_series": funding_series or [],
            "oi_series": oi_series or [],
        },
    )
    return out
