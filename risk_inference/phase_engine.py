"""
Trap Phase Engine — state machine for smart-money trap lifecycle detection.

Phases:
    NEUTRAL → ACCUMULATION → MANIPULATION → DISTRIBUTION → REVERSAL → NEUTRAL

Each phase has entry conditions and timeout guards.  The engine maintains
per-symbol state so multiple assets can be tracked concurrently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrapPhase(str, Enum):
    NEUTRAL = "NEUTRAL"
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    REVERSAL = "REVERSAL"


# Confidence assigned to each phase (increases as trap lifecycle progresses)
PHASE_CONFIDENCE: Dict[TrapPhase, float] = {
    TrapPhase.NEUTRAL: 0.00,
    TrapPhase.ACCUMULATION: 0.15,
    TrapPhase.MANIPULATION: 0.55,
    TrapPhase.DISTRIBUTION: 0.80,
    TrapPhase.REVERSAL: 0.95,
}

# Maximum bars a phase can persist before auto-resetting to NEUTRAL
MAX_PHASE_DURATION: Dict[TrapPhase, int] = {
    TrapPhase.NEUTRAL: 9999,
    TrapPhase.ACCUMULATION: 60,
    TrapPhase.MANIPULATION: 10,
    TrapPhase.DISTRIBUTION: 15,
    TrapPhase.REVERSAL: 12,
}

# Minimum bars before a phase contributes full confidence
MIN_DWELL: Dict[TrapPhase, int] = {
    TrapPhase.ACCUMULATION: 8,
    TrapPhase.MANIPULATION: 2,
    TrapPhase.DISTRIBUTION: 3,
    TrapPhase.REVERSAL: 1,
}

# Phase-aware scoring weights:
#   (structure, volume, momentum, anomaly, liquidity, retail)
PHASE_WEIGHTS: Dict[TrapPhase, Tuple[float, ...]] = {
    TrapPhase.NEUTRAL:       (0.20, 0.15, 0.15, 0.20, 0.15, 0.15),
    TrapPhase.ACCUMULATION:  (0.10, 0.25, 0.10, 0.15, 0.25, 0.15),
    TrapPhase.MANIPULATION:  (0.30, 0.15, 0.15, 0.10, 0.10, 0.20),
    TrapPhase.DISTRIBUTION:  (0.25, 0.25, 0.20, 0.10, 0.10, 0.10),
    TrapPhase.REVERSAL:      (0.15, 0.20, 0.30, 0.10, 0.15, 0.10),
}


@dataclass
class SymbolPhaseState:
    """Mutable state tracked per symbol."""
    phase: TrapPhase = TrapPhase.NEUTRAL
    phase_start_bar: int = 0
    trap_direction: Optional[str] = None  # "BULL_TRAP" or "BEAR_TRAP"
    history: List[Tuple[str, int, int]] = field(default_factory=list)


def _safe(series: pd.Series, idx: int = -1, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    try:
        return float(series.iloc[idx])
    except (IndexError, ValueError):
        return default


def _tail_mean(series: pd.Series, n: int = 5, default: float = 0.0) -> float:
    if series is None or len(series) < 1:
        return default
    return float(series.tail(n).mean())


class TrapPhaseEngine:
    """
    Maintains per-symbol trap phase state and transitions.

    Call  ``update(symbol, features_df, bar_index)``  on every new bar.
    """

    def __init__(self):
        self._states: Dict[str, SymbolPhaseState] = {}

    def _get_state(self, symbol: str) -> SymbolPhaseState:
        if symbol not in self._states:
            self._states[symbol] = SymbolPhaseState()
        return self._states[symbol]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, df: pd.DataFrame, bar_index: int) -> Dict:
        """
        Evaluate phase transitions and return current phase info.

        Returns dict with keys:
            phase, phase_confidence, trap_direction, phase_age
        """
        state = self._get_state(symbol)
        phase_age = bar_index - state.phase_start_bar

        # Timeout guard
        max_dur = MAX_PHASE_DURATION.get(state.phase, 9999)
        if phase_age > max_dur:
            self._transition(state, TrapPhase.NEUTRAL, bar_index)
            state.trap_direction = None

        # Evaluate transitions
        prev_phase = state.phase

        if state.phase == TrapPhase.NEUTRAL:
            if self._check_accumulation(df):
                self._transition(state, TrapPhase.ACCUMULATION, bar_index)

        elif state.phase == TrapPhase.ACCUMULATION:
            if self._check_manipulation(df):
                state.trap_direction = self._detect_direction(df)
                self._transition(state, TrapPhase.MANIPULATION, bar_index)
            elif not self._check_accumulation(df) and phase_age > 5:
                self._transition(state, TrapPhase.NEUTRAL, bar_index)

        elif state.phase == TrapPhase.MANIPULATION:
            if self._check_distribution(df):
                self._transition(state, TrapPhase.DISTRIBUTION, bar_index)
            elif self._breakout_holds(df, bars=3):
                # Genuine breakout — not a trap
                self._transition(state, TrapPhase.NEUTRAL, bar_index)
                state.trap_direction = None

        elif state.phase == TrapPhase.DISTRIBUTION:
            if self._check_reversal(df):
                self._transition(state, TrapPhase.REVERSAL, bar_index)
            elif self._price_resumes(df):
                self._transition(state, TrapPhase.NEUTRAL, bar_index)

        elif state.phase == TrapPhase.REVERSAL:
            if self._reversal_exhausted(df) or phase_age > 10:
                self._transition(state, TrapPhase.NEUTRAL, bar_index)
                state.trap_direction = None

        # Compute effective confidence (reduced if phase is young)
        raw_conf = PHASE_CONFIDENCE.get(state.phase, 0.0)
        current_age = bar_index - state.phase_start_bar
        min_dwell = MIN_DWELL.get(state.phase, 1)
        if current_age < min_dwell and state.phase != TrapPhase.NEUTRAL:
            raw_conf *= 0.5  # Halve confidence until established

        if state.phase != prev_phase:
            logger.info(
                "Phase transition [%s]: %s → %s (direction=%s)",
                symbol, prev_phase.value, state.phase.value, state.trap_direction,
            )

        return {
            "phase": state.phase.value,
            "phase_confidence": round(raw_conf, 3),
            "trap_direction": state.trap_direction,
            "phase_age": current_age,
        }

    def get_phase_weights(self, symbol: str) -> Tuple[float, ...]:
        """Return the 6-tuple of scoring weights for the current phase."""
        state = self._get_state(symbol)
        return PHASE_WEIGHTS.get(state.phase, PHASE_WEIGHTS[TrapPhase.NEUTRAL])

    def get_phase(self, symbol: str) -> str:
        return self._get_state(symbol).phase.value

    def get_confidence(self, symbol: str) -> float:
        return PHASE_CONFIDENCE.get(self._get_state(symbol).phase, 0.0)

    def get_trap_confidence(self, symbol: str) -> float:
        """Required public API for phase-aware scoring."""
        return self.get_confidence(symbol)

    # ------------------------------------------------------------------
    # Transition helper
    # ------------------------------------------------------------------

    def _transition(self, state: SymbolPhaseState, new_phase: TrapPhase, bar_index: int):
        state.history.append((state.phase.value, state.phase_start_bar, bar_index))
        # Keep history bounded
        if len(state.history) > 100:
            state.history = state.history[-50:]
        state.phase = new_phase
        state.phase_start_bar = bar_index

    # ------------------------------------------------------------------
    # Phase condition checks
    # ------------------------------------------------------------------

    def _check_accumulation(self, df: pd.DataFrame) -> bool:
        """Low volatility, tight range, OI building quietly, volume drying up."""
        if len(df) < 25:
            return False

        prices = df["price"].astype(float)
        volumes = df["volume"].astype(float)
        returns = prices.pct_change().fillna(0)

        volatility = returns.rolling(20, min_periods=10).std().fillna(0)
        vol_pct25 = volatility.quantile(0.25) if len(volatility.dropna()) > 10 else 0.005

        conditions = 0

        # 1. Low volatility
        if _safe(volatility) < max(vol_pct25, 0.002):
            conditions += 1

        # 2. Price flat over last 20 bars
        if abs(float(prices.tail(20).pct_change().sum())) < 0.02:
            conditions += 1

        # 3. Volume drying up (fast MA < 85% of slow MA)
        vol_fast = volumes.tail(5).mean()
        vol_slow = volumes.tail(20).mean()
        if vol_slow > 0 and vol_fast / vol_slow < 0.85:
            conditions += 1

        # 4. OI building (if available)
        if "oi_change_rate" in df.columns:
            oi_rate = _safe(df["oi_change_rate"])
            if 0 < oi_rate < 0.05:  # Quiet OI build
                conditions += 1
        else:
            # Give benefit of doubt if no OI data
            conditions += 0.5

        # 5. Range compressing
        high_range = prices.tail(20).max() - prices.tail(20).min()
        wide_range = prices.tail(60).max() - prices.tail(60).min() if len(prices) >= 60 else high_range * 2
        if wide_range > 0 and high_range / wide_range < 0.4:
            conditions += 1

        return conditions >= 3

    def _check_manipulation(self, df: pd.DataFrame) -> bool:
        """Fake breakout: price breaks S/R + FOMO/panic + rejection wick + volume spike."""
        if len(df) < 20:
            return False

        prices = df["price"].astype(float)
        volumes = df["volume"].astype(float)

        rolling_high = prices.rolling(20, min_periods=10).max()
        rolling_low = prices.rolling(20, min_periods=10).min()
        vol_ma = volumes.rolling(20, min_periods=5).mean()

        conditions_bull = 0
        conditions_bear = 0

        curr_price = _safe(prices)
        curr_high = _safe(rolling_high)
        curr_low = _safe(rolling_low)

        # Bull trap signals
        if curr_price > curr_high * 0.998:
            conditions_bull += 1
        if "fomo_index" in df.columns and _safe(df["fomo_index"]) > 0.5:
            conditions_bull += 1
        if "oi_spike_at_extreme" in df.columns and _safe(df["oi_spike_at_extreme"]) > 0:
            conditions_bull += 1
        if _safe(volumes) > _safe(vol_ma) * 1.5:
            conditions_bull += 1

        # Bear trap signals
        if curr_price < curr_low * 1.002:
            conditions_bear += 1
        if "panic_index" in df.columns and _safe(df["panic_index"]) > 0.5:
            conditions_bear += 1
        if "oi_spike_at_extreme" in df.columns and _safe(df["oi_spike_at_extreme"]) > 0:
            conditions_bear += 1
        if _safe(volumes) > _safe(vol_ma) * 1.5:
            conditions_bear += 1

        return conditions_bull >= 3 or conditions_bear >= 3

    def _check_distribution(self, df: pd.DataFrame) -> bool:
        """Price reverses back into prior range with high volume."""
        if len(df) < 10:
            return False

        prices = df["price"].astype(float)
        volumes = df["volume"].astype(float)
        returns = prices.pct_change().fillna(0)

        rolling_high = prices.rolling(20, min_periods=10).max()
        vol_ma = volumes.rolling(20, min_periods=5).mean()

        conditions = 0

        # Breakout failure (was above, now below)
        prev_above = prices.shift(1) > rolling_high.shift(1)
        now_below = prices < rolling_high
        if _safe(prev_above) and _safe(now_below):
            conditions += 1

        # High volume on reversal
        if _safe(volumes) > _safe(vol_ma) * 1.3:
            conditions += 1

        # Negative return after positive
        if _tail_mean(returns.tail(3)) < 0 < _tail_mean(returns.iloc[-6:-3]):
            conditions += 1

        # Crowd one-sided (if available)
        if "crowd_positioning_score" in df.columns:
            cps = abs(_safe(df["crowd_positioning_score"]))
            if cps > 0.4:
                conditions += 1

        return conditions >= 3

    def _check_reversal(self, df: pd.DataFrame) -> bool:
        """Sharp move opposite to the fake breakout direction."""
        if len(df) < 5:
            return False

        prices = df["price"].astype(float)
        volumes = df["volume"].astype(float)
        returns = prices.pct_change().fillna(0)
        vol_ma = volumes.rolling(20, min_periods=5).mean()

        conditions = 0

        # Fast reversal (big recent move)
        recent_move = float(returns.tail(3).sum())
        if abs(recent_move) > 0.015:
            conditions += 1

        # Volume spike on reversal
        if _safe(volumes) > _safe(vol_ma) * 2.0:
            conditions += 1

        # Near stop-loss cluster
        if "stoploss_density_below" in df.columns:
            if _safe(df["stoploss_density_below"]) > 0.6 or _safe(df["stoploss_density_above"]) > 0.6:
                conditions += 1

        # Near liquidation zone
        if "liquidation_proximity" in df.columns:
            if _safe(df["liquidation_proximity"]) > 0.5:
                conditions += 1

        return conditions >= 2

    # ------------------------------------------------------------------
    # Helper checks
    # ------------------------------------------------------------------

    def _detect_direction(self, df: pd.DataFrame) -> Optional[str]:
        prices = df["price"].astype(float)
        rolling_high = prices.rolling(20, min_periods=10).max()
        rolling_low = prices.rolling(20, min_periods=10).min()

        if _safe(prices) > _safe(rolling_high) * 0.998:
            return "BULL_TRAP"
        elif _safe(prices) < _safe(rolling_low) * 1.002:
            return "BEAR_TRAP"
        return None

    def _breakout_holds(self, df: pd.DataFrame, bars: int = 3) -> bool:
        """Return True if price has stayed above resistance for *bars* bars."""
        if len(df) < bars + 20:
            return False
        prices = df["price"].astype(float)
        rolling_high = prices.rolling(20, min_periods=10).max()
        # Check last `bars` bars are all above the level
        return all(
            float(prices.iloc[-(i + 1)]) > float(rolling_high.iloc[-(i + 2)]) * 0.998
            for i in range(bars)
        )

    def _price_resumes(self, df: pd.DataFrame) -> bool:
        """Distribution failed — price resumed breakout direction."""
        if len(df) < 5:
            return False
        prices = df["price"].astype(float)
        returns = prices.pct_change().fillna(0)
        return float(returns.tail(3).sum()) > 0.01

    def _reversal_exhausted(self, df: pd.DataFrame) -> bool:
        """Reversal is done when momentum fades and volume drops."""
        if len(df) < 5:
            return True
        prices = df["price"].astype(float)
        volumes = df["volume"].astype(float)
        returns = prices.pct_change().fillna(0)
        vol_ma = volumes.rolling(20, min_periods=5).mean()

        momentum_fading = abs(float(returns.tail(3).mean())) < 0.001
        volume_dropping = _safe(volumes) < _safe(vol_ma) * 0.8

        return momentum_fading and volume_dropping
