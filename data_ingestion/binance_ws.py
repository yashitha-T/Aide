"""
Binance WebSocket client for real-time crypto trade and ticker data.
Handles connection management, buffering, and data processing.
"""

import json
import time
import threading
import logging
from collections import deque
from typing import Dict, List, Optional, Deque

import pandas as pd
import requests
import websocket

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


class BinanceWSClient:
    """Robust Binance WebSocket client for streaming market data."""

    def __init__(self, symbols: List[str] = None):
        self.symbols = [s.lower() for s in (symbols or ["btcusdt"])]
        self.base_url = "wss://stream.binance.com:9443"
        self.tick_buffers: Dict[str, Deque[Dict]] = {s: deque(maxlen=1000) for s in self.symbols}
        self.last_price: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.last_update: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_running = False
        self.reconnect_delay = 5
        self.restricted = False
        self.last_error = None
        self.fallback_active = False
        self._fallback_thread: Optional[threading.Thread] = None

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            stream_name = data.get("stream", "")
            msg_data = data.get("data", {})

            symbol = msg_data.get("s", "").lower()
            if symbol not in self.symbols:
                return

            if "@kline_1m" in stream_name:
                kline = msg_data.get("k", {})
                taker_buy_base_asset_volume = float(kline.get("V", 0.0))
                total_volume = float(kline.get("v", 0.0))
                taker_sell_volume = max(0.0, total_volume - taker_buy_base_asset_volume)

                tick = {
                    "timestamp": float(msg_data.get("E", time.time() * 1000)) / 1000.0,
                    "price": float(kline.get("c", 0.0)),
                    "high": float(kline.get("h", 0.0)),
                    "low": float(kline.get("l", 0.0)),
                    "volume": total_volume,
                    "quote_volume": float(kline.get("q", 0.0)),
                    "taker_buy_base_asset_volume": taker_buy_base_asset_volume,
                    "total_volume": total_volume,
                    "taker_buy_volume": taker_buy_base_asset_volume,
                    "taker_sell_volume": taker_sell_volume,
                    "symbol": symbol,
                }
                self.tick_buffers[symbol].append(tick)
                self.last_price[symbol] = tick["price"]
                self.last_update[symbol] = tick["timestamp"]
                return

            if "@ticker" in stream_name:
                total_volume = float(msg_data.get("v", 0.0))
                taker_buy_base_asset_volume = float(msg_data.get("taker_buy_base_asset_volume", total_volume * 0.5))
                taker_sell_volume = max(0.0, total_volume - taker_buy_base_asset_volume)

                tick = {
                    "timestamp": float(msg_data.get("E", time.time() * 1000)) / 1000.0,
                    "price": float(msg_data.get("c", 0.0)),
                    "high": float(msg_data.get("h", 0.0)),
                    "low": float(msg_data.get("l", 0.0)),
                    "volume": total_volume,
                    "quote_volume": float(msg_data.get("q", 0.0)),
                    "taker_buy_base_asset_volume": taker_buy_base_asset_volume,
                    "total_volume": total_volume,
                    "taker_buy_volume": taker_buy_base_asset_volume,
                    "taker_sell_volume": taker_sell_volume,
                    "symbol": symbol,
                }
                self.tick_buffers[symbol].append(tick)
                self.last_price[symbol] = tick["price"]
                self.last_update[symbol] = tick["timestamp"]

        except Exception as e:
            logger.error("Error processing WS message: %s", e)

    def _on_error(self, ws, error):
        err_str = str(error)
        logger.error("WebSocket error: %s", err_str)
        self.last_error = err_str

        lower = err_str.lower()
        if "451" in err_str or "restricted location" in lower or "eligibility" in lower:
            self.restricted = True
            logger.error("Connection rejected by Binance: restricted location (HTTP 451).")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning("WebSocket closed: %s - %s", close_status_code, close_msg)
        if self.is_running:
            logger.info("Attempting to reconnect in %s seconds...", self.reconnect_delay)
            time.sleep(self.reconnect_delay)
            self._connect()

    def _on_open(self, ws):
        logger.info("WebSocket connected for symbols: %s", self.symbols)

    def _connect(self):
        streams = []
        for s in self.symbols:
            streams.append(f"{s}@ticker")
            streams.append(f"{s}@kline_1m")
        url = f"{self.base_url}/stream?streams={'/'.join(streams)}"

        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self.ws.run_forever(ping_interval=20, ping_timeout=10)

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._connect, daemon=True)
        self.thread.start()
        logger.info("Binance WS Thread started.")

    def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("Binance WS client stopped.")

    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return the buffered data for a symbol as a DataFrame."""
        symbol = symbol.lower()
        buffer = self.tick_buffers.get(symbol)
        if not buffer:
            return None

        df = pd.DataFrame(list(buffer))
        if df.empty and self.restricted and not self.fallback_active:
            self._start_fallback()
        return df

    def _start_fallback(self):
        """Start a background thread to fetch data from Yahoo Finance if Binance is blocked."""
        if self.fallback_active or not yf:
            return

        self.fallback_active = True
        self._fallback_thread = threading.Thread(target=self._fallback_loop, daemon=True)
        self._fallback_thread.start()
        logger.info("Yahoo Finance Fallback thread started.")

    def _fallback_loop(self):
        """Periodically fetch semi-live data from yfinance for all symbols."""
        while self.is_running and self.fallback_active:
            try:
                for symbol in self.symbols:
                    yf_sym = symbol.upper()
                    if "USDT" in yf_sym:
                        yf_sym = yf_sym.replace("USDT", "-USD")

                    ticker = yf.Ticker(yf_sym)
                    df = ticker.history(period="1d", interval="1m").tail(5)
                    if not df.empty:
                        for idx, row in df.iterrows():
                            total_volume = float(row["Volume"])
                            taker_buy_volume = total_volume * 0.5
                            tick = {
                                "timestamp": idx.timestamp(),
                                "price": float(row["Close"]),
                                "high": float(row["High"]),
                                "low": float(row["Low"]),
                                "volume": total_volume,
                                "taker_buy_base_asset_volume": taker_buy_volume,
                                "total_volume": total_volume,
                                "taker_buy_volume": taker_buy_volume,
                                "taker_sell_volume": max(0.0, total_volume - taker_buy_volume),
                                "symbol": symbol.lower(),
                            }
                            last_tick = self.tick_buffers[symbol.lower()][-1] if self.tick_buffers[symbol.lower()] else None
                            if not last_tick or tick["timestamp"] > last_tick["timestamp"]:
                                self.tick_buffers[symbol.lower()].append(tick)
                                self.last_price[symbol.lower()] = tick["price"]
                                self.last_update[symbol.lower()] = tick["timestamp"]

                time.sleep(15)
            except Exception as e:
                logger.error("Fallback fetch error: %s", e)
                time.sleep(30)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = BinanceWSClient(symbols=["btcusdt", "ethusdt"])
    client.start()

    try:
        while True:
            time.sleep(5)
            df = client.get_latest_data("btcusdt")
            if df is not None and not df.empty:
                print(f"Latest BTC Price: {df['price'].iloc[-1]} | Ticks: {len(df)}")
            else:
                print("Connecting...")
    except KeyboardInterrupt:
        client.stop()
