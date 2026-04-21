"""
Asset-specific parameters for trap risk features and aggregation.
Each cryptocurrency has unique volatility, liquidity, and whale behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AssetParams:
    # Feature windows and thresholds
    breakout_lookback: int
    breakout_threshold: float  # % move to consider a breakout
    absorption_window: int
    momentum_window: int
    momentum_threshold: float  # % ROC to consider exhaustion
    reversal_window: int
    compression_window: int

    # Risk aggregation weights (must sum to 1.0)
    w_structure: float
    w_volume: float
    w_momentum: float
    w_anomaly: float

    # Reason templates
    breakout_reason: str
    volume_reason: str
    momentum_reason: str

    # Derivatives / liquidity parameters (defaults so existing entries work)
    avg_leverage: float = 15.0       # Average retail leverage for liquidation estimation
    oi_spike_threshold: float = 2.0  # Std-devs for OI spike detection


ASSET_REGISTRY: Dict[str, AssetParams] = {
    "BTC-USD": AssetParams(
        breakout_lookback=20,
        breakout_threshold=0.5,
        absorption_window=10,
        momentum_window=5,
        momentum_threshold=3.0,
        reversal_window=3,
        compression_window=20,
        w_structure=0.40,
        w_volume=0.25,
        w_momentum=0.20,
        w_anomaly=0.15,
        breakout_reason="Price broke the {lookback}-bar high with declining volume and reversed within {reversal} candles.",
        volume_reason="Volume rose but price stalled over {window} bars, suggesting absorption.",
        momentum_reason="Fast {threshold}% move faded quickly, indicating exhaustion.",
    ),
    "ETH-USD": AssetParams(
        breakout_lookback=15,
        breakout_threshold=1.0,
        absorption_window=8,
        momentum_window=4,
        momentum_threshold=4.0,
        reversal_window=2,
        compression_window=15,
        w_structure=0.35,
        w_volume=0.30,
        w_momentum=0.20,
        w_anomaly=0.15,
        breakout_reason="Price broke resistance by {threshold}% without volume support and snapped back within {reversal} bars.",
        volume_reason="Volume spiked {window} bars ago but price failed to follow through.",
        momentum_reason="Momentum peaked at {threshold}% and collapsed, showing exhaustion.",
    ),
    "SOL-USD": AssetParams(
        breakout_lookback=10,
        breakout_threshold=2.0,
        absorption_window=6,
        momentum_window=3,
        momentum_threshold=6.0,
        reversal_window=2,
        compression_window=10,
        w_structure=0.30,
        w_volume=0.35,
        w_momentum=0.25,
        w_anomaly=0.10,
        breakout_reason="Price surged {threshold}% above recent highs and reversed in {reversal} bars.",
        volume_reason="Absorption signal: high volume but price compressed over {window} bars.",
        momentum_reason="Explosive {threshold}% move collapsed within {window} bars.",
    ),
    "BNB-USD": AssetParams(
        breakout_lookback=18,
        breakout_threshold=0.8,
        absorption_window=9,
        momentum_window=4,
        momentum_threshold=3.5,
        reversal_window=2,
        compression_window=18,
        w_structure=0.38,
        w_volume=0.28,
        w_momentum=0.22,
        w_anomaly=0.12,
        breakout_reason="Breakout above {lookback}-bar high faded in {reversal} bars with weak volume.",
        volume_reason="Volume rose {window} bars ago but price stalled, indicating absorption.",
        momentum_reason="Momentum peaked at {threshold}% and faded quickly.",
    ),
    "XRP-USD": AssetParams(
        breakout_lookback=12,
        breakout_threshold=1.2,
        absorption_window=7,
        momentum_window=3,
        momentum_threshold=4.5,
        reversal_window=2,
        compression_window=12,
        w_structure=0.33,
        w_volume=0.32,
        w_momentum=0.23,
        w_anomaly=0.12,
        breakout_reason="Price broke {lookback}-bar resistance and reversed within {reversal} bars.",
        volume_reason="Absorption over {window} bars: volume up, price flat.",
        momentum_reason="Fast {threshold}% move collapsed, showing exhaustion.",
    ),
}


def get_asset_params(symbol: str) -> AssetParams:
    """Get parameters for a given symbol, defaulting to BTC if unknown."""
    return ASSET_REGISTRY.get(symbol, ASSET_REGISTRY["BTC-USD"])


def supported_symbols() -> list[str]:
    """Return list of supported cryptocurrency symbols."""
    return list(ASSET_REGISTRY.keys())
