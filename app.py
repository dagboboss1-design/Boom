import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update
from openai import OpenAI

# ==========================================
# 1. DATA & ASSETS (Hybrid Live/Delayed)
# ==========================================
ASSETS = {
    # Crypto (100% LIVE via Binance/Kraken)
    "BTC/USD (Live)": "BTC/USDT",
    "ETH/USD (Live)": "ETH/USDT",
    "SOL/USD (Live)": "SOL/USDT",
    "BNB/USD (Live)": "BNB/USDT",
    "XRP/USD (Live)": "XRP/USDT",
    
    # Metals & Forex (Yahoo Finance)
    "XAU/USD (Gold)": "GC=F",
    "XAG/USD (Silver)": "SI=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "DXY (Dollar Index)": "DX-Y.NYB"
}

TF_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"
}

def fetch_crypto_live(symbol, timeframe):
    """Fetches REAL-TIME tick data from Binance, falls back to Kraken if blocked."""
    exchanges = ["binance", "kraken", "coinbase"]
    for ex_name in exchanges:
        try:
            ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("ts")
        except Exception:
            continue # Try the next exchange if Binance is blocked
    raise ValueError(f"Could not fetch live crypto data for {symbol} (Network blocked all exchanges)")

def fetch_yahoo(symbol, timeframe):
    """Fetches Forex/Metals from Yahoo Finance."""
    tf_map = {"1m": ("7d", "1m"), "5m": ("1mo", "5m"), "15m": ("1mo", "15m"), 
              "1h": ("3mo", "1h"), "4h": ("1y", "1h"), "1d": ("2y", "1d")}
    period, interval = tf_map.get(timeframe, ("1mo", "15m"))
    
    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data found for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
        
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if timeframe == "4h":
        df = df.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", 
            "close": "last", "volume": "sum"
        }).dropna()
    return df.dropna()

def fetch_data(symbol, timeframe):
    if "/" in symbol:
        return fetch_crypto_live(symbol, timeframe)
    return fetch_yahoo(symbol, timeframe)

# ==========================================
# 2. INDICATORS & MATH
# ==========================================
def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = sma + 2*std
    df["bb_lower"] = sma - 2*std
    
    macd_line = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_sig"] = macd_line.ewm(span=9, adjust=False).mean()
    
    return df.dropna()

def generate_signal(df):
    if len(df) < 60:
        return {"signal": "WAIT", "notes": "Not enough data"}
    
    last = df.iloc[-1]
    bull = bear = 0
    notes = []
    
    if last["ema20"] > last["ema50"]: bull += 1; notes.append("EMA20>50")
    else: bear += 1; notes.append("EMA20<50")
    
    if 50 < last["rsi"] < 70: bull += 1
    elif 30 < last["rsi"] < 50: bear += 1
    
    if last["macd"] > last["macd_sig"]: bull += 1; notes.append("MACD Bull")
    else: bear += 1; notes.append("MACD Bear")
    
    sig = "BUY" if bull >= 2 and bull > bear else "SELL" if bear >= 2 and bear > bull else "HOLD"
    color = "#00d47e" if sig == "BUY" else "#ff4d4d" if sig == "SELL" else "#ffcc00"
    
    return {
        "signal": sig, "color": color, 
        "price": round(float(last["close"]), 2) if last["close"] > 10 else round(float(last["close"]), 4),
        "rsi": round(float(last["rsi"]), 2),
        "notes": " · ".join(notes)
    }

# ==========================================
# 3. PLOTLY CHART
# ==========================================
def build_chart(df, symbol, selected_indicators):
    has_rsi = "RSI" in selected_indicators
    has_macd = "MACD" in selected_indicators
    
    rows = 1 + int(has_rsi) + int(has_macd)
    heights = [0.65]
    if rows == 2: heights = [0.75, 0.25]
    elif rows == 3: heights = [0.60, 0.20, 0.20]
    
    titles = [""] + (["RSI"] if has_rsi else []) + (["MACD"] if has_macd else [])
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                        row_heights=heights, vertical_spacing=0.03, subplot_titles=titles)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#00d47e", decreasing_line_color="#ff4d4d"
    ), row=1, col=1)
    
    if "EMA 20" in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20", line=dict(color="#4da3ff", width=1)), row=1, col=1)
    if "EMA 50" in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50", line=dict(color="#ffaa33", width=1)), row=1, col=1)
    if "Bollinger Bands" in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="#666", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="#666", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(120,120,120,0.05)"), row=1, col=1)

    current_row = 2
    if has_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#f59e0b", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#555", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#555", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1
        
    if has_macd:
        fig.add_trace(go.Bar(x=df.index, y=df["macd"]-df["macd_sig"], name="Hist", marker_color="#888"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#4da3ff", width=1)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"], name="Signal", line=dict(color="#ffaa33", width=1)), row=current_row, col=1)

    fig.update_layout(
        template="plotly_dark", height=600 if rows == 1 else 750,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, uirevision="keep", # Keeps zoom
        paper_bgcolor="#131722", plot_bgcolor="#131722",
        legend=dict(orientation="h", y=1.02, x=0)
    )
    fig.update_xaxes(gridcolor="#2a2e39", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2e39", zeroline=False)
    return fig

# ==========================================
# 4. AI CHAT (OPENAI ONLY)
# ==========================================
def ask_ai(message, symbol, timeframe, df, api_key):
    if not api_key:
        return "⚠️ **OPENAI_API_KEY is missing.** Go to Railway -> Variables, add your key, and redeploy."
    
    try:
        client = OpenAI(api_key=api_key)
        latest = df.iloc[-1]
        
        context = (
            f"Asset: {symbol} | Timeframe: {timeframe}\n"
            f"Current Price: {latest['close']:.4f}\n"
            f"RSI: {latest['rsi']:.2f}\n"
            f"EMA20: {latest['ema20']:.4f} | EMA50: {latest['ema50']:.4f}\n"
            f"MACD: {latest['macd']:.4f} (Signal: {latest['macd_sig']:.4f})\n"
        )
        
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a professional financial AI assistant. Analyze the provided market data. Do not invent prices. Be concise. End with 'Not financial advice.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

# ==========================================
# 5. UI & DASHBOARD
# ==========================================
app = Dash(__name__, title="Pro Terminal")
app.layout = html.Div([
    html.Div([
        html.H2("📈 PRO TERMINAL", style={"margin": "0", "color": "#00d47e", "fontWeight": "bold"}),
        html.P("Live Crypto · Forex · AI Analysis", style={"margin": "0", "color": "#888", "fontSize": "12px"})
    ], style={"padding": "15px 20px", "backgroundColor": "#131722", "borderBottom": "1px solid #2a2e39"}),
    
    html.Div([
        html.Div([
            html.Label("Asset", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="asset-dropdown",
                options=[{"label": k, "value": v} for k, v in ASSETS.items()],
                value="BTC/USDT", clearable=False, style={"color": "#000"}
            )
        ], style={"flex": "1", "minWidth": "150px"}),
        
        html.Div([
            html.Label("Timeframe", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="tf-dropdown",
                options=[{"label": k, "value": k} for k in TF_MAP.keys()],
                value="5m", clearable=False, style={"color": "#000"}
            )
        ], style={"flex": "1", "minWidth": "100px"}),
        
        html.Div([
            html.Label("Indicators", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(
                id="indicator-dropdown",
                options=[{"label": i, "value": i} for i in ["EMA 20", "EMA 50", "Bollinger Bands", "RSI", "MACD"]],
                value=["EMA 20", "EMA 50", "RSI"], multi=True, style={"color": "#000"}
            )
        ], style={"flex": "2", "minWidth": "250px"})
    ], style={"display": "flex", "gap": "15px", "padding": "15px 20px", "backgroundColor": "#1e222d", "flexWrap": "wrap"}),

    html.Div(id="signal-box", style={"padding": "10px 20px", "backgroundColor": "#131722", "borderBottom": "1px solid #2a2e39"}),
    
    html.Div([
        html.Div([
            dcc.Graph(id="main-chart", config={"displayModeBar": False}),
            dcc.Interval(id="refresh-timer", interval=5000) # 5 SECONDS LIVE REFRESH
        ], style={"flex": "2", "minWidth": "300px"}),
        
        html.Div([
            html.Div("🤖 AI Assistant", style={"padding": "10px", "borderBottom": "1px solid #2a2e39", "fontWeight": "bold", "color": "#00d47e"}),
            html.Div(id="chat-log", style={"flex": "1", "overflowY": "auto", "padding": "15px", "display": "flex", "flexDirection": "column", "gap": "10px"}),
            html.Div([
                dcc.Input(id="chat-input", type="text", placeholder="Ask about this chart...", n_submit=0,
                          style={"flex": "1", "padding": "10px", "borderRadius": "5px", "border": "1px solid #2a2e39", "backgroundColor": "#1e222d", "color": "#fff"}),
                html.Button("Send", id="chat-btn", n_clicks=0, style={"padding": "10px 15px", "backgroundColor": "#00d47e", "color": "#000", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontWeight": "bold"})
            ], style={"display": "flex", "gap": "10px", "padding": "10px", "borderTop": "1px solid #2a2e39", "backgroundColor": "#131722"})
        ], style={"flex": "1", "minWidth": "300px", "backgroundColor": "#131722", "display": "flex", "flexDirection": "column", "borderLeft": "1px solid #2a2e39"})
    ], style={"display": "flex", "flexWrap": "wrap", "height": "calc(100vh - 200px)"})

], style={"backgroundColor": "#131722", "color": "#d1d4dc", "fontFamily": "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, sans-serif", "minHeight": "100vh"})

@app.callback(
    Output("main-chart", "figure"),
    Output("signal-box", "children"),
    Input("asset-dropdown", "value"),
    Input("tf-dropdown", "value"),
    Input("indicator-dropdown", "value"),
    Input("refresh-timer", "n_intervals")
)
def update_chart(asset, tf, indicators, _n):
    try:
        df = fetch_data(asset, tf)
        df = add_indicators(df)
        sig = generate_signal(df)
        fig = build_chart(df, asset, indicators or [])
        
        info = html.Div([
            html.Span(f"{asset} · {tf} · ", style={"color": "#aaa"}),
            html.Span(f"${sig['price']}", style={"fontSize": "20px", "fontWeight": "bold", "color": "#fff"}),
            html.Span(f" | SIGNAL: {sig['signal']}", style={"color": sig['color'], "fontWeight": "bold", "marginLeft": "15px"}),
            html.Span(f" | RSI: {sig['rsi']}", style={"color": "#aaa", "marginLeft": "10px"}),
            html.Span(f" | {sig['notes']}", style={"color": "#666", "fontSize": "12px", "marginLeft": "10px"})
        ])
        return fig, info
    except Exception as e:
        return go.Figure(), html.Div(f"Error: {e}", style={"color": "#ff4d4d"})

@app.callback(
    Output("chat-log", "children"),
    Output("chat-input", "value"),
    Input("chat-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-log", "children"),
    State("asset-dropdown", "value"),
    State("tf-dropdown", "value"),
    prevent_initial_call=True
)
def chat(_n, _sub, text, log, asset, tf):
    if not text: return no_update, no_update
    log = list(log) if isinstance(log, list) else []
    
    log.append(html.Div(text, style={"alignSelf": "flex-end", "backgroundColor": "#2962ff", "color": "#fff", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px"}))
    
    try:
        df = add_indicators(fetch_data(asset, tf))
        reply = ask_ai(text, asset, tf, df, os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        reply = f"Error loading data: {e}"
        
    log.append(html.Div(reply, style={"alignSelf": "flex-start", "backgroundColor": "#2a2e39", "color": "#d1d4dc", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px", "whiteSpace": "pre-wrap"}))
    return log, ""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
