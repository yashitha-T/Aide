"""Historical trap label generation for MarketTrap evaluation."""

from __future__ import annotations

import pandas as pd


def generate_trap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate binary trap labels using breakout-followthrough failure rules.

    Rules:
    - breakout_up: price > rolling_high_20
    - breakout_down: price < rolling_low_20
    - bull trap: breakout_up and price drops >= 3% within next 10 bars
    - bear trap: breakout_down and price rises >= 3% within next 10 bars

    Output:
    - trap_label in {0, 1}
    """
    if df is None or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        out["trap_label"] = []
        return out

    out = df.copy()
    if "price" not in out.columns:
        if "close" in out.columns:
            out["price"] = out["close"]
        else:
            raise ValueError("generate_trap_labels requires 'price' or 'close' column")

    prices = out["price"].astype(float)

    rolling_high_20 = prices.rolling(window=20, min_periods=20).max().shift(1)
    rolling_low_20 = prices.rolling(window=20, min_periods=20).min().shift(1)

    breakout_up = prices > rolling_high_20
    breakout_down = prices < rolling_low_20

    labels = []
    n = len(out)
    for i in range(n):
        current_price = float(prices.iloc[i])
        if i + 1 >= n:
            labels.append(0)
            continue

        horizon_end = min(n, i + 11)
        future_prices = prices.iloc[i + 1 : horizon_end]
        if future_prices.empty:
            labels.append(0)
            continue

        min_future = float(future_prices.min())
        max_future = float(future_prices.max())

        bull_trap = bool(breakout_up.iloc[i]) and (((current_price - min_future) / current_price) >= 0.03)
        bear_trap = bool(breakout_down.iloc[i]) and (((max_future - current_price) / current_price) >= 0.03)

        labels.append(1 if (bull_trap or bear_trap) else 0)

    out["trap_label"] = labels
    out["trap_label"] = out["trap_label"].fillna(0).astype(int)
    return out
