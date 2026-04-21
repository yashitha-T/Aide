"""
Binance Futures REST client for derivatives intelligence.

Provides direct polling helpers:
- get_open_interest(symbol)
- get_funding_rate(symbol)
- get_long_short_ratio(symbol)

And a background client that polls these endpoints every 15-30 seconds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_FAPI_BASE = "https://fapi.binance.com"


def _request_json(endpoint: str, params: Optional[Dict] = None, timeout: float = 5.0):
    """Shared Binance Futures GET helper."""
    url = f"{_FAPI_BASE}{endpoint}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_open_interest(symbol: str) -> Dict:
    """Fetch latest open interest for a symbol (e.g. BTCUSDT)."""
    symbol = symbol.upper()
    payload = _request_json("/fapi/v1/openInterest", params={"symbol": symbol})
    return {
        "timestamp": time.time(),
        "symbol": symbol,
        "open_interest": float(payload.get("openInterest", 0.0)),
    }


def get_funding_rate(symbol: str) -> Dict:
    """Fetch latest funding rate for a symbol."""
    symbol = symbol.upper()
    payload = _request_json("/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 1})
    entry = payload[-1] if isinstance(payload, list) and payload else {}
    funding_ts = float(entry.get("fundingTime", time.time() * 1000)) / 1000.0
    return {
        "timestamp": funding_ts,
        "symbol": symbol,
        "funding_rate": float(entry.get("fundingRate", 0.0)),
    }


def get_long_short_ratio(symbol: str) -> Dict:
    """Fetch latest global long/short account ratio for a symbol."""
    symbol = symbol.upper()
    payload = _request_json(
        "/futures/data/globalLongShortAccountRatio",
        params={"symbol": symbol, "period": "5m", "limit": 1},
    )
    entry = payload[-1] if isinstance(payload, list) and payload else {}
    ratio_ts = float(entry.get("timestamp", time.time() * 1000)) / 1000.0
    return {
        "timestamp": ratio_ts,
        "symbol": symbol,
        "long_short_ratio": float(entry.get("longShortRatio", 1.0)),
        "long_account": float(entry.get("longAccount", 0.0)),
        "short_account": float(entry.get("shortAccount", 0.0)),
    }


class BinanceFuturesClient:
    """Poll Binance Futures REST endpoints in a background loop."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        poll_interval: float = 20.0,
        buffer_size: int = 500,
    ):
        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT"])]
        # Enforce requested polling cadence (15-30s)
        self.poll_interval = float(min(30.0, max(15.0, poll_interval)))
        self.buffer_size = int(buffer_size)

        self.oi_buffers: Dict[str, Deque[Dict]] = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.funding_buffers: Dict[str, Deque[Dict]] = {s: deque(maxlen=buffer_size) for s in self.symbols}
        self.ls_ratio_buffers: Dict[str, Deque[Dict]] = {s: deque(maxlen=buffer_size) for s in self.symbols}

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.restricted = False
        self.last_error: Optional[str] = None

    def latest_oi(self, symbol: str) -> Optional[Dict]:
        buf = self.oi_buffers.get(symbol.upper())
        return buf[-1] if buf else None

    def latest_funding(self, symbol: str) -> Optional[Dict]:
        buf = self.funding_buffers.get(symbol.upper())
        return buf[-1] if buf else None

    def latest_ls_ratio(self, symbol: str) -> Optional[Dict]:
        buf = self.ls_ratio_buffers.get(symbol.upper())
        return buf[-1] if buf else None

    def get_oi_series(self, symbol: str, n: int = 60) -> List[Dict]:
        return list(self.oi_buffers.get(symbol.upper(), deque()))[-n:]

    def get_funding_series(self, symbol: str, n: int = 30) -> List[Dict]:
        return list(self.funding_buffers.get(symbol.upper(), deque()))[-n:]

    def get_ls_ratio_series(self, symbol: str, n: int = 60) -> List[Dict]:
        return list(self.ls_ratio_buffers.get(symbol.upper(), deque()))[-n:]

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("BinanceFuturesClient started (poll every %.1fs)", self.poll_interval)

    def stop(self):
        self.is_running = False
        logger.info("BinanceFuturesClient stopped")

    def _poll_loop(self):
        while self.is_running:
            for symbol in self.symbols:
                self._fetch_symbol(symbol)
            time.sleep(self.poll_interval)

    def _fetch_symbol(self, symbol: str):
        try:
            self.oi_buffers[symbol].append(get_open_interest(symbol))
        except Exception as exc:
            self._track_error(exc)

        try:
            self.funding_buffers[symbol].append(get_funding_rate(symbol))
        except Exception as exc:
            self._track_error(exc)

        try:
            self.ls_ratio_buffers[symbol].append(get_long_short_ratio(symbol))
        except Exception as exc:
            self._track_error(exc)

    def _track_error(self, exc: Exception):
        self.last_error = str(exc)
        err_lower = self.last_error.lower()
        if "451" in self.last_error or "restricted" in err_lower or "eligibility" in err_lower:
            self.restricted = True
        logger.debug("Binance futures request failed: %s", self.last_error)
