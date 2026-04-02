"""
MarketTrap Professional - Institutional-Grade Crypto Intelligence Terminal
Professional real-time dashboard with Binance WebSocket integration and advanced risk detection.
"""

from collections import deque
from datetime import datetime
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.binance_ws import BinanceWSClient
from risk_inference.engine import MarketTrapEngine

logger = logging.getLogger(__name__)

# --- Configuration & Styling ---
st.set_page_config(
    page_title="MarketTrap Pro Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .terminal-header.alert {
        border-bottom: 1px solid rgba(248, 81, 73, 0.6);
        box-shadow: 0 0 18px rgba(248, 81, 73, 0.35);
    }
    .status-dot.alert {
        background-color: var(--bearish);
        box-shadow: 0 0 12px var(--bearish);
    }
    :root {
        --bg-color: #05070a;
        --panel-bg: #0d1117;
        --border-color: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-blue: #58a6ff;
        --bullish: #23d18b;
        --bearish: #f85149;
        --warning: #d29922;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    .terminal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.5rem;
        background: var(--panel-bg);
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        letter-spacing: -0.5px;
        color: var(--accent-blue);
    }
    .terminal-status {
        font-size: 0.8rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--bullish);
        box-shadow: 0 0 10px var(--bullish);
        animation: pulse 2s infinite;
    }

    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(13, 17, 23, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(48, 54, 61, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.15);
    }
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .risk-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(15px);
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(48, 54, 61, 0.5);
        height: 100%;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .trap-type-badge {
        margin-top: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-radius: 999px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        background: #11161d;
        padding: 0.35rem 0.8rem;
        text-align: center;
    }

    .control-chip {
        margin-top: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        text-align: center;
        width: 100%;
    }

    .reasons-panel {
        width: 100%;
        margin-top: 0.9rem;
        background: #0b1117;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.8rem;
    }

    .attribution-panel {
        width: 100%;
        margin-top: 0.7rem;
        background: #0b1117;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 0.7rem;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    [data-testid="stSidebar"] {
        background-color: var(--panel-bg);
        border-right: 1px solid var(--border-color);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

CRITICAL_THRESHOLD = 70.0
CRITICAL_STREAK_UPDATES = 3
MAX_TRAP_EVENTS = 10
EMA_ALPHA = 0.35


# --- State Management ---
if "binance_client" not in st.session_state:
    st.session_state.binance_client = BinanceWSClient(symbols=["btcusdt", "ethusdt", "solusdt", "bnbusdt"])
    st.session_state.binance_client.start()

if "engine" not in st.session_state:
    st.session_state.engine = MarketTrapEngine()

if "terminal_events" not in st.session_state:
    st.session_state.terminal_events = deque(maxlen=30)
    # Add initial startup message
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    st.session_state.terminal_events.appendleft(f'<span style="color:#8b949e">[{timestamp}]</span> <span style="color:#23d18b">SYSTEM ONLINE: MONITORING LIVE FEED...</span>')

if "last_control" not in st.session_state:
    st.session_state.last_control = ""

if "risk_state" not in st.session_state:
    st.session_state.risk_state = {}

if "trap_history" not in st.session_state:
    st.session_state.trap_history = deque(maxlen=MAX_TRAP_EVENTS)

if "sim_mode_announced" not in st.session_state:
    st.session_state.sim_mode_announced = False

if "last_risk_log_level" not in st.session_state:
    st.session_state.last_risk_log_level = None

client = st.session_state.binance_client
engine = st.session_state.engine

def log_event(message: str, type: str = "info"):
    timestamp = datetime.utcnow().strftime("%H:%M:%S")
    color = {"info": "#58a6ff", "warning": "#d29922", "error": "#f85149", "success": "#23d18b"}.get(type, "#8b949e")
    st.session_state.terminal_events.appendleft(f'<span style="color:#8b949e">[{timestamp}]</span> <span style="color:{color}">{message.upper()}</span>')

# If the Binance WS client detected a geographic restriction (HTTP 451), show a clear banner
if getattr(client, 'restricted', False):
    st.error(
        "Live Binance feed appears to be blocked from this host (HTTP 451 - restricted location).\n"
        "Using fallback/historical data. To get live ticks, run the WS client from an allowed host or proxy the feed from an allowed region."
    )
    # store flag in session for other UI logic
    st.session_state['ws_restricted'] = True
else:
    st.session_state['ws_restricted'] = False


# --- UI Components ---
def render_header(is_critical: bool = False):
    now = datetime.utcnow().strftime("%H:%M:%S UTC")
    header_class = "terminal-header alert" if is_critical else "terminal-header"
    dot_class = "status-dot alert" if is_critical else "status-dot"
    status_text = "CRITICAL RISK" if is_critical else "LIVE FEED"

    st.markdown(
        f"""
    <div class="{header_class}">
        <div class="terminal-title">MARKET TRAP DETECTION</div>
        <div class="terminal-status">
            <span>{status_text}</span>
            <div class="{dot_class}"></div>
            <span style="margin-left: 10px; font-family: 'JetBrains Mono'">{now}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metrics(symbol: str, context_df: Optional[pd.DataFrame] = None):
    df = client.get_latest_data(symbol)
    if df is not None and not df.empty:
        curr = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else curr

        change = curr["price"] - prev["price"]
        change_pct = (change / prev["price"] * 100) if prev["price"] != 0 else 0
        color = "var(--bullish)" if change >= 0 else "var(--bearish)"

        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Price ({symbol.upper()})</div>
                <div class="metric-value">${curr['price']:,.2f}</div>
                <div class="metric-delta" style="color: {color}">
                    {"+" if change >= 0 else ""}{change:.2f} ({change_pct:+.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">24h High</div>
                <div class="metric-value">${curr['high']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">24h Low</div>
                <div class="metric-value">${curr['low']:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume</div>
                <div class="metric-value">{curr['volume']:,.2f}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return curr

    # Fallback metrics from merged 1m context (historical/simulated path).
    if context_df is not None and not context_df.empty:
        curr = context_df.iloc[-1]
        prev = context_df.iloc[-2] if len(context_df) > 1 else curr
        change = float(curr["price"]) - float(prev["price"])
        change_pct = (change / float(prev["price"]) * 100) if float(prev["price"]) != 0 else 0
        color = "var(--bullish)" if change >= 0 else "var(--bearish)"
        context_high = float(context_df["price"].tail(120).max())
        context_low = float(context_df["price"].tail(120).min())
        context_volume = float(curr["volume"])

        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-label">Price ({symbol.upper()})</div>
                <div class="metric-value">${float(curr['price']):,.2f}</div>
                <div class="metric-delta" style="color: {color}">
                    {"+" if change >= 0 else ""}{change:.2f} ({change_pct:+.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Context High (120m)</div>
                <div class="metric-value">${context_high:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Context Low (120m)</div>
                <div class="metric-value">${context_low:,.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volume (1m)</div>
                <div class="metric-value">{context_volume:,.2f}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return curr

    return None


@st.cache_data(ttl=60)
def fetch_historical_context(symbol: str, limit: int = 120):
    """Fetch recent historical 1-minute data from Binance REST API."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["timestamp"] = df["timestamp"] / 1000.0
            df["price"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df[["timestamp", "price", "volume"]]
    except Exception as exc:
        logger.warning("Historical context fetch failed for %s: %s", symbol, exc)
    return pd.DataFrame()


def build_merged_1m_frame(symbol: str):
    rt_df = client.get_latest_data(symbol)
    hist_df = fetch_historical_context(symbol, limit=120)
    frames = []

    if hist_df is not None and not hist_df.empty:
        hist = hist_df[["timestamp", "price", "volume"]].copy()
        hist["timestamp"] = hist["timestamp"].astype(int)
        hist["source_rank"] = 0
        frames.append(hist)

    if rt_df is not None and not rt_df.empty:
        rt_df = rt_df.copy()
        rt_df["datetime"] = pd.to_datetime(rt_df["timestamp"], unit="s")
        rt_df["volume_delta"] = rt_df["volume"].diff().fillna(0).clip(lower=0)
        rt_1m = (
            rt_df.set_index("datetime")
            .resample("1min")
            .agg({"price": "last", "volume_delta": "sum"})
            .dropna()
            .reset_index()
        )
        if not rt_1m.empty:
            rt_1m["timestamp"] = (rt_1m["datetime"].astype("int64") // 10**9).astype(int)
            rt_1m = rt_1m.rename(columns={"volume_delta": "volume"})
            rt_1m["source_rank"] = 1
            frames.append(rt_1m[["timestamp", "price", "volume", "source_rank"]])

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.sort_values(["timestamp", "source_rank"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .drop(columns=["source_rank"])
        .sort_values("timestamp")
        .tail(180)
    )
    merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="s")
    return merged


def create_advanced_chart(df_1m: pd.DataFrame):
    if df_1m is None or df_1m.empty:
        st.info("Awaiting more ticks for visualization...")
        return

    frame = df_1m.copy()
    
    # Calculate Indicators
    frame['ema9'] = frame['price'].ewm(span=9, adjust=False).mean()
    frame['ema21'] = frame['price'].ewm(span=21, adjust=False).mean()
    
    # Simple VWAP proxy for 1m session
    v = frame['volume'].replace(0, 1)
    frame['vwap'] = (frame['price'] * v).cumsum() / v.cumsum()
    
    frame["datetime"] = frame["datetime"] + pd.to_timedelta(frame.index, unit="ms")
    frame["volume_plot"] = frame["volume"].clip(upper=frame["volume"].quantile(0.95))

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])

    # Price Line
    fig.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=frame["price"],
            mode="lines",
            line=dict(color="#58a6ff", width=2.5),
            name="Price",
        ),
        secondary_y=True,
    )

    # EMA 9
    fig.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=frame['ema9'],
            mode="lines",
            line=dict(color="rgba(35, 209, 139, 0.4)", width=1, dash='dot'),
            name="EMA 9",
        ),
        secondary_y=True,
    )
    
    # EMA 21
    fig.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=frame['ema21'],
            mode="lines",
            line=dict(color="rgba(248, 81, 73, 0.4)", width=1, dash='dot'),
            name="EMA 21",
        ),
        secondary_y=True,
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(
            x=frame["datetime"],
            y=frame['vwap'],
            mode="lines",
            line=dict(color="rgba(210, 153, 34, 0.6)", width=1.5),
            name="VWAP",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=frame["datetime"],
            y=frame["volume_plot"],
            marker=dict(color="rgba(48, 54, 61, 0.3)"),
            width=1000 * 60,
            name="Volume",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", family="JetBrains Mono"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(showticklabels=False, secondary_y=False, showgrid=False)
    fig.update_yaxes(side="right", secondary_y=True, showgrid=True, gridcolor="#161b22")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="markettrap_main_price_volume_chart",
    )


def render_risk_gauge(risk_score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "TRAP PROBABILITY", "font": {"size": 14, "color": "#8b949e", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#30363d"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#0d1117",
                "borderwidth": 2,
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 30], "color": "rgba(35, 209, 139, 0.1)"},
                    {"range": [30, 70], "color": "rgba(210, 153, 34, 0.1)"},
                    {"range": [70, 100], "color": "rgba(248, 81, 73, 0.1)"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "JetBrains Mono"},
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="markettrap_main_risk_gauge",
    )


@st.cache_data(ttl=2, show_spinner=False)
def fetch_order_book_depth(symbol: str, limit: int = 10):
    url = "https://api.binance.com/api/v3/depth"
    try:
        res = requests.get(url, params={"symbol": symbol.upper(), "limit": limit}, timeout=4)
        if res.status_code != 200:
            return None
        payload = res.json()
        bids = [(float(price), float(qty)) for price, qty in payload.get("bids", [])]
        asks = [(float(price), float(qty)) for price, qty in payload.get("asks", [])]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks}
    except Exception as exc:
        logger.debug("Order book fetch failed for %s: %s", symbol, exc)
        return None


def render_order_book(symbol: str, curr_price: float):
    depth = fetch_order_book_depth(symbol, limit=10)
    if depth is None:
        st.markdown(
            """
            <div style="background:#0d1117; border:1px solid var(--border-color); border-radius:8px; padding:0.8rem; height:100%;">
                <div style="color:var(--text-secondary); font-size:0.7rem; letter-spacing:1px; margin-bottom:0.5rem;">ORDER DEPTH</div>
                <div style="font-size:0.78rem; color:var(--text-secondary);">Order book unavailable from REST endpoint.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    bids = depth["bids"][:10]
    asks = depth["asks"][:10]

    html = f'<div style="background:#0d1117; border: 1px solid var(--border-color); border-radius: 8px; padding: 0.8rem; height: 100%;">'
    html += f'<div style="color:var(--text-secondary); font-size:0.7rem; letter-spacing:1px; margin-bottom:0.5rem;">ORDER DEPTH (REST)</div>'
    
    # Asks (Sellers)
    for p, v in reversed(asks):
        fill = min(100, int(v * 40))
        html += (f'<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-family:\'JetBrains Mono\'; position:relative; margin-bottom:2px;">'
                 f'<span style="color:var(--bearish); z-index:1;">{p:,.2f}</span>'
                 f'<span style="z-index:1;">{v:.3f}</span>'
                 f'<div style="position:absolute; right:0; top:0; bottom:0; width:{fill}%; background:rgba(248,81,73,0.15); border-radius:2px;"></div></div>')

    html += f'<div style="text-align:center; padding:0.4rem; font-size:1rem; font-weight:700; color:var(--text-primary); margin:0.3rem 0; border-top:1px solid #30363d; border-bottom:1px solid #30363d;">{curr_price:,.2f}</div>'

    # Bids (Buyers)
    for p, v in bids:
        fill = min(100, int(v * 40))
        html += (f'<div style="display:flex; justify-content:space-between; font-size:0.7rem; font-family:\'JetBrains Mono\'; position:relative; margin-bottom:2px;">'
                 f'<span style="color:var(--bullish); z-index:1;">{p:,.2f}</span>'
                 f'<span style="z-index:1;">{v:.3f}</span>'
                 f'<div style="position:absolute; right:0; top:0; bottom:0; width:{fill}%; background:rgba(35,209,139,0.15); border-radius:2px;"></div></div>')
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_terminal_log():
    html = f'<div style="background:#05070a; border: 1px solid var(--border-color); border-radius: 8px; padding: 0.8rem; height: 280px; overflow-y: auto; font-family:\'JetBrains Mono\'; font-size:0.72rem; line-height:1.4;">'
    html += f'<div style="color:var(--text-secondary); margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:1px;">System Execution Log</div>'
    
    if st.session_state.terminal_events:
        for event in st.session_state.terminal_events:
            html += f'<div>{event}</div>'
    else:
        html += f'<div style="color:var(--text-secondary)">AWAITING DATA BLOCKS...</div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def update_risk_state(symbol: str, raw_risk_score: float):
    # Keep risk stable: smooth short-lived spikes before threshold checks.
    state = st.session_state.risk_state.setdefault(
        symbol,
        {"streak": 0, "last_smoothed_risk": 0.0, "smoothed_risk": 0.0},
    )

    if state["smoothed_risk"] == 0.0:
        smoothed_risk = raw_risk_score
    else:
        smoothed_risk = (EMA_ALPHA * raw_risk_score) + ((1 - EMA_ALPHA) * state["smoothed_risk"])

    if smoothed_risk >= CRITICAL_THRESHOLD:
        state["streak"] += 1
    else:
        state["streak"] = 0

    crossed_above = state["last_smoothed_risk"] < CRITICAL_THRESHOLD <= smoothed_risk
    state["last_smoothed_risk"] = smoothed_risk
    state["smoothed_risk"] = smoothed_risk
    return state["streak"] >= CRITICAL_STREAK_UPDATES, crossed_above, state["streak"], round(smoothed_risk, 1)


def render_reasons_panel(reasons, risk_score):
    st.markdown('<div class="reasons-panel"><div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.45rem;">TOP TRAP REASONS</div>', unsafe_allow_html=True)
    
    if reasons and risk_score >= 40:
        for reason in reasons:
            conf_color = "var(--text-secondary)"
            if reason['confidence'] > 50: conf_color = "white"
            st.markdown(f"<div style='font-size:0.78rem; margin-bottom:0.28rem;'>• {reason['reason']} <span style='color:{conf_color}'>({reason['confidence']:.1f}%)</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.78rem;color:var(--text-secondary);'>Awaiting elevated trap conditions...</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_control_indicator(control_state: str):
    if control_state == "Buyers in Control":
        bg = "rgba(35, 209, 139, 0.12)"
        fg = "var(--bullish)"
    elif control_state == "Sellers in Control":
        bg = "rgba(248, 81, 73, 0.12)"
        fg = "var(--bearish)"
    else:
        bg = "rgba(210, 153, 34, 0.12)"
        fg = "var(--warning)"

    st.markdown(
        f"<div class='control-chip' style='background:{bg}; color:{fg};'>ORDERFLOW BIAS: {control_state.upper()}</div>",
        unsafe_allow_html=True,
    )


def render_component_attribution(components: dict):
    labels = [
        ("Structure", components.get("structure_failure", 0.0)),
        ("Volume", components.get("volume_behavior", 0.0)),
        ("Momentum", components.get("momentum_exhaustion", 0.0)),
        ("Anomaly", components.get("anomaly", 0.0)),
    ]
    total = sum(max(0.0, val) for _, val in labels)
    if total <= 0:
        total = 1.0

    st.markdown(
        '<div class="attribution-panel"><div style="font-size:0.70rem;color:var(--text-secondary);margin-bottom:0.45rem;">TRAP DNA (COMPONENT WEIGHT SHARE)</div>',
        unsafe_allow_html=True,
    )
    for label, value in labels:
        share = (max(0.0, value) / total) * 100.0
        st.markdown(
            f"""
            <div style="font-size:0.72rem; display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                <span>{label}</span><span style="color:var(--text-secondary)">{share:.1f}%</span>
            </div>
            <div style="height:5px; border-radius:6px; background:#1f2937; margin-bottom:0.42rem;">
                <div style="height:5px; border-radius:6px; width:{share:.1f}%; background:#58a6ff;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# Sidebar Settings
with st.sidebar:
    st.markdown("### TERMINAL SETTINGS")
    symbol_choice = st.selectbox("ACTIVE SYMBOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"], index=0).lower()
    refresh_rate = st.slider("SCRAPE INTERVAL (S)", 1, 5, 2)
    st.markdown("---")
    st.markdown("### SYSTEM LOG")
    if engine.model:
        st.success("Anomaly Model: LOADED")
    else:
        st.warning("Anomaly Model: MISSING")

    st.caption(f"Critical trigger rule: {CRITICAL_STREAK_UPDATES} consecutive updates above {CRITICAL_THRESHOLD:.0f}%.")

    if st.button("RESTART FEED"):
        client.stop()
        client.start()
        st.rerun()
        
    if st.button("CLEAR LOG"):
        st.session_state.terminal_events.clear()
        st.rerun()

# Data + Snapshot
# If the WS client detected a restriction, or no live ticks are available,
# fall back to historical REST data or simulated data so the UI always shows something.
def _make_simulated_1m(symbol: str, periods: int = 180):
    np.random.seed(42)
    base_price = {
        'btcusdt': 43000,
        'ethusdt': 2600,
        'solusdt': 105,
        'bnbusdt': 310,
    }.get(symbol.lower(), 1000)
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq='1min')
    price = np.cumprod(1 + np.random.randn(periods) * 0.001) * base_price
    volume = np.random.randint(100000, 5000000, size=periods)
    df = pd.DataFrame({
        'datetime': timestamps,
        'price': price,
        'volume': volume,
    })
    df['timestamp'] = (df['datetime'].astype('int64') // 10**9).astype(int)
    return df


data_mode = "live"
if st.session_state.get('ws_restricted', False):
    st.info('Live Binance feed blocked — showing historical/simulated data instead.')
    hist = fetch_historical_context(symbol_choice, limit=180)
    if hist is None or hist.empty:
        merged_1m = _make_simulated_1m(symbol_choice, periods=180)
        data_mode = "simulated"
    else:
        merged_1m = hist.copy()
        merged_1m['datetime'] = pd.to_datetime(merged_1m['timestamp'], unit='s')
        data_mode = "historical"
else:
    merged_1m = build_merged_1m_frame(symbol_choice)
    if merged_1m is None or merged_1m.empty:
        # Try REST historical first, then simulated
        hist = fetch_historical_context(symbol_choice, limit=180)
        if hist is None or hist.empty:
            merged_1m = _make_simulated_1m(symbol_choice, periods=180)
            data_mode = "simulated"
        else:
            merged_1m = hist.copy()
            merged_1m['datetime'] = pd.to_datetime(merged_1m['timestamp'], unit='s')
            data_mode = "historical"

snapshot = engine.get_risk_snapshot(symbol_choice, merged_1m)
snapshot["main_reason"] = snapshot["reasons"][0]["reason"] if snapshot["reasons"] else "Monitoring..."
snapshot["buyer_seller_control"] = snapshot["control"]

is_critical, crossed_above_70, streak_count, smoothed_risk = update_risk_state(symbol_choice, snapshot["risk_score"])

if data_mode == "simulated":
    # Prevent simulated snapshots from polluting real alert streak state.
    st.session_state.risk_state[symbol_choice] = {"streak": 0, "last_smoothed_risk": 0.0, "smoothed_risk": 0.0}
    is_critical = False
    crossed_above_70 = False
    streak_count = 0
    smoothed_risk = round(snapshot["risk_score"], 1)
    if not st.session_state.sim_mode_announced:
        log_event("SIMULATION MODE ACTIVE: ALERTING/HISTORY SUPPRESSED", "warning")
        st.session_state.sim_mode_announced = True
else:
    st.session_state.sim_mode_announced = False
    risk_level_for_log = "CRITICAL" if smoothed_risk >= CRITICAL_THRESHOLD else ("ELEVATED" if smoothed_risk >= 40 else "NORMAL")

    if crossed_above_70:
        log_event(f"TRAP DETECTED: {snapshot['trap_type']} ({snapshot['risk_score']}%)", "error")
        st.toast(f"🚨 CRITICAL TRAP: {snapshot['trap_type']}", icon="⚠️")
    elif smoothed_risk > 40 and st.session_state.last_risk_log_level != "ELEVATED":
        log_event(f"ELEVATED RISK: {snapshot['trap_type']} ({smoothed_risk}%)", "warning")

    if snapshot["control"] != st.session_state.last_control:
        log_event(f"BIAS SHIFT: {snapshot['control'].upper()}", "info")
        st.session_state.last_control = snapshot["control"]
    elif not st.session_state.last_control:  # First run
        log_event(f"INITIAL BIAS: {snapshot['control'].upper()}", "info")
        st.session_state.last_control = snapshot["control"]

    st.session_state.last_risk_log_level = risk_level_for_log

render_header(is_critical=is_critical)

curr_data = render_metrics(symbol_choice, merged_1m)

if data_mode != "simulated" and crossed_above_70 and curr_data is not None:
    st.session_state.trap_history.appendleft(
        {
            "Time (UTC)": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Symbol": symbol_choice.upper(),
            "Price": round(float(curr_data["price"]), 2),
            "Trap Risk %": round(smoothed_risk, 1),
            "Trap Type": snapshot["trap_type"],
            "Main Reason": snapshot["main_reason"],
        }
    )

col_chart, col_risk = st.columns([0.7, 0.3])

with col_chart:
    chart_slot = st.empty()
    with chart_slot.container():
        st.markdown(
            """
        <div style="background: var(--panel-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem;">
            <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 1rem; font-family: 'JetBrains Mono'">REAL-TIME TAPE</div>
        """,
            unsafe_allow_html=True,
        )
        create_advanced_chart(merged_1m)
        st.markdown("</div>", unsafe_allow_html=True)

with col_risk:
    risk_score = smoothed_risk

    risk_slot = st.empty()
    with risk_slot.container():
        st.markdown('<div class="risk-gauge-container">', unsafe_allow_html=True)
        render_risk_gauge(risk_score)

    risk_level = "LOW"
    risk_color = "var(--bullish)"
    if is_critical:
        risk_level = "CRITICAL"
        risk_color = "var(--bearish)"
    elif risk_score > 30:
        risk_level = "ELEVATED"
        risk_color = "var(--warning)"

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 1rem; width:100%;">
                <div style="color: var(--text-secondary); font-size: 0.8rem;">STATUS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color}; letter-spacing: 2px;">{risk_level}</div>
                <div style="font-size: 0.68rem; color: var(--text-secondary); margin-top: 8px;">
                    STREAK ABOVE 70%: {streak_count}/{CRITICAL_STREAK_UPDATES}
                </div>
                <div class="trap-type-badge">TRAP TYPE: {snapshot['trap_type']}</div>
                <div style="margin-top: 8px; font-size: 0.62rem; color: var(--text-secondary);">DATA MODE: {data_mode.upper()}</div>
                <div style="margin-top: 10px; font-size: 0.6rem; color: var(--text-secondary);">SENTIMENT: {"🐂 BULLISH" if snapshot['control'] == "Buyers in Control" else "🐻 BEARISH" if snapshot['control'] == "Sellers in Control" else "⚖️ NEUTRAL"}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        render_control_indicator(snapshot["buyer_seller_control"])
        reasons_to_show = snapshot["reasons"] if risk_score >= 40 else []
        render_reasons_panel(reasons_to_show, risk_score)
        render_component_attribution(snapshot.get("components", {}))
        st.markdown("</div>", unsafe_allow_html=True)

col_log, col_depth = st.columns([0.7, 0.3])
with col_log:
    render_terminal_log()
with col_depth:
    if curr_data is not None:
        render_order_book(symbol_choice, curr_data["price"])
    else:
        st.info("Awaiting price for depth...")

st.markdown("### TRAP HISTORY (LAST 10)")
if st.session_state.trap_history:
    history_df = pd.DataFrame(list(st.session_state.trap_history))
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No trap events have crossed above 70% yet.")

# Real-time update loop
time.sleep(refresh_rate)
st.rerun()
