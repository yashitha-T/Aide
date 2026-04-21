"""Liquidity intelligence features for MarketTrap."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _find_swing_lows(prices: pd.Series, order: int = 5) -> pd.Series:
    lows = pd.Series(False, index=prices.index)
    for i in range(order, len(prices) - order):
        window = prices.iloc[i - order : i + order + 1]
        if prices.iloc[i] == window.min():
            lows.iloc[i] = True
    return lows


def _find_swing_highs(prices: pd.Series, order: int = 5) -> pd.Series:
    highs = pd.Series(False, index=prices.index)
    for i in range(order, len(prices) - order):
        window = prices.iloc[i - order : i + order + 1]
        if prices.iloc[i] == window.max():
            highs.iloc[i] = True
    return highs


def compute_stoploss_density(
    df: pd.DataFrame,
    price_col: str = "price",
    lookback: int = 60,
    range_pct: float = 0.02,
    decay_lambda: float = 0.05,
) -> pd.DataFrame:
    out = df.copy()
    prices = out[price_col].astype(float)

    swing_low_mask = _find_swing_lows(prices, order=5)
    swing_high_mask = _find_swing_highs(prices, order=5)

    density_below = np.zeros(len(prices))
    density_above = np.zeros(len(prices))

    for i in range(lookback, len(prices)):
        current_price = prices.iloc[i]
        lower_bound = current_price * (1 - range_pct)
        upper_bound = current_price * (1 + range_pct)

        below = 0.0
        above = 0.0
        for j in range(max(0, i - lookback), i):
            age_weight = np.exp(-decay_lambda * (i - j))
            level = prices.iloc[j]
            if swing_low_mask.iloc[j] and lower_bound <= level <= current_price:
                below += age_weight
            if swing_high_mask.iloc[j] and current_price <= level <= upper_bound:
                above += age_weight

        density_below[i] = below
        density_above[i] = above

    for col, values in [
        ("stoploss_density_below", density_below),
        ("stoploss_density_above", density_above),
    ]:
        s = pd.Series(values, index=out.index)
        rolling_max = s.rolling(window=lookback, min_periods=1).max().replace(0, 1)
        out[col] = (s / rolling_max).clip(0, 1).fillna(0)

    return out


def compute_round_number_proximity(
    price: float | pd.Series | pd.DataFrame,
    price_col: str = "price",
    round_intervals: Optional[List[float]] = None,
):
    """
    Computes round-number proximity.

    Accepts:
    - scalar float price -> returns float
    - pd.Series of prices -> returns pd.Series
    - pd.DataFrame -> returns DataFrame with `round_number_proximity`
    """
    if round_intervals is None:
        round_intervals = [1000, 500, 100, 50, 10, 5, 1, 0.5]

    def _prox(p: float) -> float:
        p = float(p)
        if p <= 0:
            return 0.0
        min_dist = min(abs(p - (round(p / interval) * interval)) for interval in round_intervals)
        return float(max(0.0, 1.0 - min_dist / (0.01 * p)))

    if isinstance(price, pd.DataFrame):
        out = price.copy()
        out["round_number_proximity"] = out[price_col].astype(float).apply(_prox).clip(0, 1)
        return out
    if isinstance(price, pd.Series):
        return price.astype(float).apply(_prox).clip(0, 1)
    return float(np.clip(_prox(float(price)), 0, 1))


def compute_liquidation_proximity(
    df: pd.DataFrame,
    oi_data: Optional[List[Dict]] = None,
    price_col: str = "price",
    lookback: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    prices = out[price_col].astype(float)

    avg_leverage = 15.0
    if oi_data and len(oi_data) >= 5:
        oi_vals = [float(x.get("open_interest", 0.0)) for x in oi_data[-20:] if x.get("open_interest") is not None]
        if len(oi_vals) >= 2:
            oi_growth = (oi_vals[-1] - oi_vals[0]) / max(abs(oi_vals[0]), 1.0)
            avg_leverage = float(np.clip(12.0 + (oi_growth * 30.0), 8.0, 25.0))

    liq_offset = 1.0 / avg_leverage
    rolling_high = prices.rolling(window=lookback, min_periods=1).max()
    rolling_low = prices.rolling(window=lookback, min_periods=1).min()

    long_liq_level = rolling_high * (1 - liq_offset)
    short_liq_level = rolling_low * (1 + liq_offset)

    sigma = prices * 0.005
    long_prox = np.exp(-0.5 * ((prices - long_liq_level) / sigma.replace(0, np.nan)) ** 2)
    short_prox = np.exp(-0.5 * ((prices - short_liq_level) / sigma.replace(0, np.nan)) ** 2)

    out["liquidation_proximity"] = np.maximum(long_prox, short_prox)
    out["liquidation_proximity"] = out["liquidation_proximity"].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
    return out


def compute_oi_features(df: pd.DataFrame, oi_series: List[Dict], price_col: str = "price") -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    if not oi_series:
        out["oi_change_rate"] = 0.0
        out["oi_price_divergence"] = 0.0
        out["oi_spike_at_extreme"] = 0
        return out

    oi_vals = [float(x.get("open_interest", 0.0)) for x in oi_series]
    oi_arr = np.array(oi_vals[-n:]) if len(oi_vals) >= n else np.array(oi_vals)
    if len(oi_arr) < n:
        pad_val = oi_arr[0] if len(oi_arr) > 0 else 0.0
        oi_arr = np.concatenate([np.full(n - len(oi_arr), pad_val), oi_arr])

    oi_series_pd = pd.Series(oi_arr, index=out.index)
    oi_prev = oi_series_pd.shift(1).replace(0, np.nan)
    out["oi_change_rate"] = ((oi_series_pd - oi_prev) / oi_prev).replace([np.inf, -np.inf], 0).fillna(0)

    prices = out[price_col].astype(float)
    price_return = prices.pct_change().fillna(0)
    divergence = pd.Series(0.0, index=out.index)
    divergence[(out["oi_change_rate"] > 0.01) & (price_return.abs() < 0.002)] = 1.0
    divergence[(out["oi_change_rate"] < -0.01) & (price_return.abs() > 0.005)] = -1.0
    out["oi_price_divergence"] = divergence

    rolling_high = prices.rolling(window=20, min_periods=5).max()
    rolling_low = prices.rolling(window=20, min_periods=5).min()
    at_extreme = (prices >= rolling_high * 0.995) | (prices <= rolling_low * 1.005)
    oi_std = out["oi_change_rate"].rolling(window=20, min_periods=5).std().fillna(0.01)
    oi_spike = out["oi_change_rate"].abs() > 2 * oi_std
    out["oi_spike_at_extreme"] = (at_extreme & oi_spike).astype(int)

    return out


def compute_funding_features(df: pd.DataFrame, funding_series: List[Dict]) -> pd.DataFrame:
    out = df.copy()
    out["funding_rate"] = float(funding_series[-1].get("funding_rate", 0.0)) if funding_series else 0.0
    return out


def compute_ls_ratio_features(df: pd.DataFrame, ls_series: List[Dict]) -> pd.DataFrame:
    out = df.copy()
    out["long_short_ratio"] = float(ls_series[-1].get("long_short_ratio", 1.0)) if ls_series else 1.0
    return out


def compute_all_liquidity_features(
    df: pd.DataFrame,
    oi_series: Optional[List[Dict]] = None,
    funding_series: Optional[List[Dict]] = None,
    ls_series: Optional[List[Dict]] = None,
    price_col: str = "price",
) -> pd.DataFrame:
    out = compute_stoploss_density(df, price_col=price_col)
    out = compute_round_number_proximity(out, price_col=price_col)
    out = compute_liquidation_proximity(out, oi_data=oi_series or [], price_col=price_col)
    out = compute_oi_features(out, oi_series or [], price_col=price_col)
    out = compute_funding_features(out, funding_series or [])
    out = compute_ls_ratio_features(out, ls_series or [])
    return out
