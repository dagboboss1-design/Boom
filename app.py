import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update

# ==========================================
# 1. DATA & MATH
# ==========================================
ASSETS = {
    "BTC/USDT": "BTC/USDT", "ETH/USDT": "ETH/USDT", "SOL/USDT": "SOL/USDT",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "EUR/USD": "EURUSD=X", 
    "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "GBP/JPY": "GBPJPY=X", "DXY": "DX-Y.NYB"
}

def fetch_data(symbol, timeframe):
    if "/" in symbol:
        for ex_name in ["binance", "kraken", "coinbase"]:
            try:
                ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df.set_index("ts")
            except Exception:
                continue
        raise ValueError(f"Could not fetch crypto for {symbol}")
    else:
        tf_map = {"1m": ("7d", "1m"), "5m": ("1mo", "5m"), "15m": ("1mo", "15m"), 
                  "1h": ("3mo", "1h"), "4h": ("1y", "1h"), "1d": ("2y", "1d")}
        period, interval = tf_map.get(timeframe, ("1mo", "15m"))
        raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if raw.empty: raise ValueError(f"No data for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        if timeframe == "4h":
            df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return df.dropna()

def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    return df.dropna()

def build_chart(df, symbol):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#00d47e", decreasing_line_color="#ff4d4d"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20", line=dict(color="#4da3ff", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50", line=dict(color="#ffaa33", width=1)), row=1, col=1)
    colors = ["#00d47e" if c >= o else "#ff4d4d" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False, paper_bgcolor="#131722", plot_bgcolor="#131722")
    return fig

# ==========================================
# 2. AI LOGIC (DIRECT POLLINATIONS API)
# ==========================================
def ask_free_ai(user_msg):
    system_prompt = (
        "You are a trading AI. You can show charts and analyze markets. "
        "If the user wants a chart, reply EXACTLY with this JSON format on a new line: "
        '{"action": "chart", "symbol": "BTC/USDT", "timeframe": "1h"} \n'
        "Available symbols: BTC/USDT, ETH/USDT, SOL/USDT, XAU/USD, XAG/USD, EUR/USD, GBP/USD, USD/JPY, GBP/JPY, DXY. "
        "Available timeframes: 5m, 15m, 1h, 4h, 1d. "
        "If they want analysis, reply EXACTLY with: "
        '{"action": "analyze", "symbol": "BTC/USDT", "timeframe": "1h"} \n'
        "Otherwise, just answer normally. Never invent prices."
    )

    payload = {
        "model": "openai",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.3
    }

    try:
        # Call the Free Pollinations API (No API Key Required!)
        response = requests.post("https://text.pollinations.ai/openai", json=payload, timeout=30)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        
        charts = []
        final_text = reply
        
        # Check if the AI triggered a command
        if '{"action":' in reply:
            try:
                json_str = reply.split('{', 1)[1].split('}')[0]
                cmd = json.loads('{' + json_str + '}')
                
                sym = cmd.get("symbol")
                tf = cmd.get("timeframe")
                action = cmd.get("action")
                
                if action == "chart":
                    df = add_indicators(fetch_data(ASSETS[sym], tf))
                    charts.append(dcc.Graph(figure=build_chart(df, sym), config={"displayModeBar": False}, style={"height": "400px", "marginTop": "10px"}))
                    final_text = f"Here is the live chart for **{sym}** on the **{tf}** timeframe."
                    
                elif action == "analyze":
                    df = add_indicators(fetch_data(ASSETS[sym], tf))
                    last = df.iloc[-1]
                    final_text = f"**{sym} Analysis ({tf})**\n\n- **Price:** {last['close']:.4f}\n- **RSI:** {last['rsi']:.2f}\n- **EMA20:** {last['ema20']:.4f}\n- **EMA50:** {last['ema50']:.4f}\n\n*Not financial advice.*"
            except Exception as e:
                final_text = f"I tried to fetch the chart/data, but failed: {e}"
                
        return final_text, charts

    except Exception as e:
        return f"⚠️ Pollinations API Error: {e}", []

# ==========================================
# 3. UI
# ==========================================
app = Dash(__name__, title="Free AI Trader")
app.layout = html.Div([
    html.Div([
        html.H2("🤖 Free AI Trader", style={"margin": "0", "color": "#00d47e", "fontWeight": "bold"}),
        html.P("Powered by Pollinations.ai · No API Key Needed", style={"margin": "0", "color": "#888", "fontSize": "12px"})
    ], style={"padding": "15px 20px", "backgroundColor": "#131722", "borderBottom": "1px solid #2a2e39"}),
    
    html.Div(id="chat-log", style={
        "flex": "1", "overflowY": "auto", "padding": "20px", "display": "flex", "flexDirection": "column", "gap": "15px"
    }, children=[
        html.Div("👋 Try: 'Draw me the Bitcoin chart on 1h' or 'Analyze Gold'", 
                 style={"color": "#888", "fontSize": "14px", "textAlign": "center", "marginTop": "20px"})
    ]),
    
    html.Div([
        dcc.Input(id="chat-input", type="text", placeholder="Ask the AI...", n_submit=0,
                  style={"flex": "1", "padding": "15px", "borderRadius": "8px", "border": "1px solid #2a2e39", "backgroundColor": "#1e222d", "color": "#fff", "fontSize": "14px"}),
        html.Button("Send", id="chat-btn", n_clicks=0, style={"padding": "15px 30px", "backgroundColor": "#00d47e", "color": "#000", "border": "none", "borderRadius": "8px", "cursor": "pointer", "fontWeight": "bold", "fontSize": "14px"})
    ], style={"display": "flex", "gap": "10px", "padding": "20px", "borderTop": "1px solid #2a2e39", "backgroundColor": "#131722"})
], style={"backgroundColor": "#131722", "color": "#d1d4dc", "fontFamily": "system-ui, sans-serif", "height": "100vh", "display": "flex", "flexDirection": "column", "margin": "0"})

@app.callback(
    Output("chat-log", "children"),
    Output("chat-input", "value"),
    Input("chat-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-log", "children"),
    prevent_initial_call=True
)
def chat(_n, _sub, text, log):
    if not text: return no_update, no_update
    log = list(log) if isinstance(log, list) else []
    
    if len(log) == 1 and isinstance(log[0], dict) and "👋 Try" in str(log[0]): log = []
    
    log.append(html.Div(text, style={"alignSelf": "flex-end", "backgroundColor": "#2962ff", "color": "#fff", "padding": "12px 16px", "borderRadius": "15px 15px 2px 15px", "maxWidth": "80%", "fontSize": "14px"}))
    
    reply_text, charts = ask_free_ai(text)
    ai_children = [html.Div(reply_text, style={"whiteSpace": "pre-wrap"})] + charts
    
    log.append(html.Div(ai_children, style={"alignSelf": "flex-start", "backgroundColor": "#2a2e39", "color": "#d1d4dc", "padding": "12px 16px", "borderRadius": "15px 15px 15px 2px", "maxWidth": "85%", "fontSize": "14px"}))
    
    return log, ""

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
