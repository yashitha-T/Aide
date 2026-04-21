"""Realtime trap intelligence helpers for MarketTrap dashboard."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_latest(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    return float(series.iloc[-1])


def _compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_component_scores(df_1m: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return normalized component scores (5 components) and diagnostics."""
    if df_1m is None or len(df_1m) < 25:
        zero = {
            "structure_failure": 0.0,
            "volume_behavior": 0.0,
            "momentum_exhaustion": 0.0,
            "liquidity_intelligence": 0.0,
            "retail_behavior": 0.0,
        }
        return zero, {}

    frame = df_1m.copy()
    frame["returns"] = frame["price"].pct_change().fillna(0.0)
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    rolling_high = frame["price"].rolling(20, min_periods=5).max().shift(1)
    breakout = frame["price"] > rolling_high
    breakout_prev = breakout.shift(1)
    breakout_prev = breakout_prev.where(breakout_prev.notna(), False).astype(bool)
    breakout_failure = breakout_prev & (frame["price"] < rolling_high)

    near_high = frame["price"] >= frame["price"].rolling(20, min_periods=5).max() * 0.995
    low_rel_volume = frame["volume"] < frame["volume"].rolling(20, min_periods=5).mean() * 0.9

    rsi = _compute_rsi(frame["price"], period=14)
    rsi_falling_from_high = (rsi.shift(1) > 68) & (rsi < rsi.shift(1) - 2)

    momentum_fade = (
        frame["returns"].rolling(3, min_periods=3).mean()
        < frame["returns"].rolling(8, min_periods=5).mean()
    )

    price_up_volume_down = (frame["returns"] > 0) & (frame["volume_change"] < 0)

    structure_failure = _clip01(
        0.7 * float(breakout_failure.tail(5).mean())
        + 0.3 * float(((frame["returns"] < -0.002) & near_high).tail(5).mean())
    )
    volume_behavior = _clip01(
        0.65 * float(price_up_volume_down.tail(8).mean())
        + 0.35 * float((near_high & low_rel_volume).tail(8).mean())
    )
    momentum_exhaustion = _clip01(
        0.6 * float(rsi_falling_from_high.tail(8).mean())
        + 0.4 * float(momentum_fade.tail(8).mean())
    )

    # --- Liquidity Intelligence Component ---
    # Reads columns added by feature_engineering/liquidity_features.py
    sl_below = _safe_latest(frame.get("stoploss_density_below", pd.Series(dtype=float)))
    sl_above = _safe_latest(frame.get("stoploss_density_above", pd.Series(dtype=float)))
    oi_div = abs(_safe_latest(frame.get("oi_price_divergence", pd.Series(dtype=float))))
    liq_prox = _safe_latest(frame.get("liquidation_proximity", pd.Series(dtype=float)))
    fr_raw = _safe_latest(frame.get("funding_rate", pd.Series(dtype=float)))
    rnd_prox = _safe_latest(frame.get("round_number_proximity", pd.Series(dtype=float)))
    # Funding rate z-score (rough): typical range ±0.001
    fr_zscore = _clip01(abs(fr_raw) / 0.001) if fr_raw != 0 else 0.0

    liquidity_intelligence = _clip01(
        0.30 * max(sl_below, sl_above)
        + 0.25 * oi_div
        + 0.20 * liq_prox
        + 0.15 * fr_zscore
        + 0.10 * rnd_prox
    )

    # --- Retail Behavior Component ---
    # Reads columns added by feature_engineering/retail_features.py
    fomo_res = _safe_latest(frame.get("fomo_at_resistance", pd.Series(dtype=float)))
    panic_sup = _safe_latest(frame.get("panic_at_support", pd.Series(dtype=float)))
    crowd = abs(_safe_latest(frame.get("crowd_positioning_score", pd.Series(dtype=float))))
    tbr = _safe_latest(frame.get("taker_buy_ratio", pd.Series(dtype=float)), default=0.5)
    # Extreme imbalance = retail-driven
    taker_imbalance = abs(tbr - 0.5) * 2.0  # Scale [0, 1]

    retail_behavior = _clip01(
        0.40 * max(fomo_res, panic_sup)
        + 0.30 * crowd
        + 0.30 * taker_imbalance
    )

    diagnostics = {
        "breakout_failure_strength": float(breakout_failure.tail(8).mean()),
        "price_up_volume_down_strength": float(price_up_volume_down.tail(8).mean()),
        "rsi_fall_strength": float(rsi_falling_from_high.tail(8).mean()),
        "near_high_low_volume_strength": float((near_high & low_rel_volume).tail(8).mean()),
        "momentum_fade_strength": float(momentum_fade.tail(8).mean()),
        "latest_rsi": _safe_latest(rsi, 50.0),
        "latest_return": _safe_latest(frame["returns"], 0.0),
        "latest_volume_change": _safe_latest(frame["volume_change"], 0.0),
        # New diagnostics
        "liquidity_stoploss_density": max(sl_below, sl_above),
        "liquidity_oi_divergence": oi_div,
        "liquidity_proximity": liq_prox,
        "retail_fomo_at_resistance": fomo_res,
        "retail_panic_at_support": panic_sup,
        "retail_crowd_score": crowd,
    }

    components = {
        "structure_failure": structure_failure,
        "volume_behavior": volume_behavior,
        "momentum_exhaustion": momentum_exhaustion,
        "liquidity_intelligence": liquidity_intelligence,
        "retail_behavior": retail_behavior,
    }
    return components, diagnostics


def classify_trap_type(
    components: Dict[str, float],
    anomaly_component: float,
    phase_direction: str | None = None,
) -> str:
    """Classify the dominant trap type from component scores.

    Primarily driven by component dominance to ensure trap_type changes
    as market conditions shift. Phase direction is informational only.
    """
    # Find dominant component
    dominant = max(
        {
            "structure_failure": components.get("structure_failure", 0.0),
            "volume_behavior": components.get("volume_behavior", 0.0),
            "momentum_exhaustion": components.get("momentum_exhaustion", 0.0),
            "liquidity_intelligence": components.get("liquidity_intelligence", 0.0),
            "retail_behavior": components.get("retail_behavior", 0.0),
            "anomaly": anomaly_component,
        }.items(),
        key=lambda item: item[1],
    )[0]

    labels = {
        "structure_failure": "Breakout Failure Trap",
        "volume_behavior": "Distribution Trap",
        "momentum_exhaustion": "Fake Momentum Trap",
        "liquidity_intelligence": "Liquidity Sweep Trap",
        "retail_behavior": "Retail Herding Trap",
        "anomaly": "Anomaly-Driven Trap",
    }
    
    trap_type = labels.get(dominant, "Fake Momentum Trap")
    
    # If phase direction suggests a bias, annotate it (but don't override component-based classification)
    if phase_direction == "BULL_TRAP" and components.get("volume_behavior", 0) > 0.3:
        trap_type = f"{trap_type} [Bull Trap Bias]"
    elif phase_direction == "BEAR_TRAP" and components.get("retail_behavior", 0) > 0.3:
        trap_type = f"{trap_type} [Bear Trap Bias]"
    
    return trap_type


def extract_trap_reasons(
    components: Dict[str, float],
    diagnostics: Dict[str, float],
    anomaly_component: float,
    max_reasons: int = 3,
) -> List[Dict[str, float]]:
    candidates: List[Tuple[str, float]] = []

    breakout_conf = max(
        components.get("structure_failure", 0.0),
        diagnostics.get("breakout_failure_strength", 0.0),
    )
    candidates.append(("Price broke resistance but failed to hold", breakout_conf))

    volume_conf = max(
        components.get("volume_behavior", 0.0),
        diagnostics.get("price_up_volume_down_strength", 0.0),
    )
    candidates.append(("Price rising without volume support", volume_conf))

    momentum_conf = max(
        components.get("momentum_exhaustion", 0.0),
        diagnostics.get("rsi_fall_strength", 0.0),
    )
    candidates.append(("Momentum exhausted after sharp push", momentum_conf))

    candidates.append(("Anomalous behavior detected vs recent history", anomaly_component))

    # --- New: liquidity and retail reasons ---
    liq_conf = components.get("liquidity_intelligence", 0.0)
    if liq_conf > 0.15:
        sl_density = diagnostics.get("liquidity_stoploss_density", 0.0)
        liq_prox = diagnostics.get("liquidity_proximity", 0.0)
        if sl_density > liq_prox:
            candidates.append(("Price near stop-loss cluster — liquidity hunt likely", liq_conf))
        else:
            candidates.append(("Price near estimated liquidation zone", liq_conf))

    retail_conf = components.get("retail_behavior", 0.0)
    if retail_conf > 0.15:
        fomo = diagnostics.get("retail_fomo_at_resistance", 0.0)
        panic = diagnostics.get("retail_panic_at_support", 0.0)
        if fomo > panic:
            candidates.append(("Retail FOMO detected at resistance — bull trap setup", retail_conf))
        else:
            candidates.append(("Retail panic at support — bear trap setup", retail_conf))

    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    filtered = [item for item in ranked if item[1] >= 0.20]
    if len(filtered) < 2:
        filtered = ranked[:2]
    candidates = filtered[:max_reasons]
    return [
        {"reason": reason, "confidence": round(_clip01(conf) * 100, 1)}
        for reason, conf in candidates
    ]


def buyer_seller_control(df_1m: pd.DataFrame) -> str:
    if df_1m is None or len(df_1m) < 10:
        return "Neutral"

    frame = df_1m.copy()
    frame["returns"] = frame["price"].pct_change().fillna(0.0)
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    price_trend = float(frame["returns"].tail(5).mean())
    vol_trend = float(frame["volume_change"].tail(5).mean())

    # Integrate taker buy ratio for higher-fidelity control detection
    tbr = 0.5
    if "taker_buy_ratio" in frame.columns:
        tbr = float(frame["taker_buy_ratio"].tail(5).mean())

    buy_signal = price_trend > 0.0008 and vol_trend >= -0.01 and tbr >= 0.48
    sell_signal = price_trend < -0.0008 and vol_trend >= -0.01 and tbr <= 0.52

    if buy_signal:
        return "Buyers in Control"
    if sell_signal:
        return "Sellers in Control"
    return "Neutral"


def compute_directional_bias_from_components(
    components: Dict[str, float],
    risk_score: float,
) -> Tuple[str, str]:
    """
    Compute directional bias (LONG/SHORT/NEUTRAL) from dominant signal component.
    
    Returns: (bias, confidence_level)
    - bias: "LONG", "SHORT", or "NEUTRAL"
    - confidence_level: "LOW", "MEDIUM", or "HIGH" based on dominance strength
    
    Logic:
    1. Identify dominant component (highest score)
    2. Map dominant component to direction:
       - volume_behavior, structure_failure, momentum_exhaustion, liquidity_intelligence → SHORT
       - retail_behavior (high panic) → LONG (reversal setup)
    3. Assess dominance: if max_component >> others → HIGH confidence
       if max_component > others → MEDIUM confidence
       otherwise → LOW confidence with directional bias if any signal exists
    """
    if not components or all(v == 0.0 for v in components.values()):
        return "NEUTRAL", "LOW"
    
    # Extract component values
    vol_behavior = float(components.get("volume_behavior", 0.0))
    struct_failure = float(components.get("structure_failure", 0.0))
    momentum_exhaust = float(components.get("momentum_exhaustion", 0.0))
    liquidity_intel = float(components.get("liquidity_intelligence", 0.0))
    retail_behavior = float(components.get("retail_behavior", 0.0))
    
    # Find max and second-max to assess dominance
    scores = [vol_behavior, struct_failure, momentum_exhaust, liquidity_intel, retail_behavior]
    sorted_scores = sorted(scores, reverse=True)
    max_score = sorted_scores[0]
    second_max_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    
    # Early exit if all components very low
    if max_score < 0.1:
        return "NEUTRAL", "LOW"
    
    # Determine dominance level based on gap between max and second-max
    dominance_gap = max_score - second_max_score
    
    if dominance_gap > 0.25:  # Clear dominance
        dominance_confidence = "HIGH"
    elif dominance_gap > 0.10:  # Moderate dominance
        dominance_confidence = "MEDIUM"
    else:  # Weak dominance
        dominance_confidence = "LOW"
    
    # Verify with risk_score alignment
    if risk_score >= 60:
        confidence_level = "HIGH"
    elif risk_score >= 40:
        confidence_level = "MEDIUM" if dominance_confidence != "LOW" else "LOW"
    elif risk_score >= 25 and dominance_confidence == "HIGH":
        confidence_level = "MEDIUM"
    elif risk_score >= 15 and dominance_confidence in ("HIGH", "MEDIUM"):
        confidence_level = "LOW"
    else:
        confidence_level = "LOW"
    
    # Determine bias from dominant component
    if max_score == vol_behavior:
        # High volume absorption → SHORT (distribution trap)
        bias = "SHORT"
    elif max_score == struct_failure:
        # Breakout failed → SHORT (bull trap)
        bias = "SHORT"
    elif max_score == momentum_exhaust:
        # Momentum fading → SHORT (trend fade)
        bias = "SHORT"
    elif max_score == liquidity_intel:
        # Liquidation pressure → SHORT (liquidity trap)
        bias = "SHORT"
    elif max_score == retail_behavior:
        # Retail herding/panic → LONG (reversal opportunity)
        bias = "LONG"
    else:
        bias = "NEUTRAL"
    
    # Ensure we always return a directional bias when signals exist
    if bias == "NEUTRAL" and max_score > 0.15:
        # Fallback: if highest is retail → LONG, else → SHORT
        bias = "LONG" if max_score == retail_behavior else "SHORT"
    
    return bias, confidence_level
