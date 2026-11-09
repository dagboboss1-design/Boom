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
# 1. DATA & ASSETS
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
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, uirevision="keep", paper_bgcolor="#131722", plot_bgcolor="#131722")
    fig.update_xaxes(gridcolor="#2a2e39", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2e39", zeroline=False)
    return fig

# ==========================================
# 2. AI LOGIC (GROQ + POLLINATIONS FALLBACK)
# ==========================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "change_chart",
            "description": "Change the asset and timeframe displayed on the dashboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": list(ASSETS.keys())},
                    "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]}
                },
                "required": ["asset", "timeframe"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market",
            "description": "Fetch live data and indicators for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": list(ASSETS.keys())},
                    "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d"]}
                },
                "required": ["asset", "timeframe"]
            }
        }
    }
]

def run_ai_agent(user_msg, current_asset, current_tf):
    groq_key = os.getenv("GROQ_API_KEY")
    poll_key = os.getenv("POLLINATIONS_API_KEY", "free") # Pollinations works with "free"
    
    if not groq_key and not poll_key:
        return "⚠️ Both GROQ_API_KEY and POLLINATIONS_API_KEY are missing.", no_update, no_update

    messages = [
        {"role": "system", "content": "You are an AI trading assistant. Use tools to fetch data or change the chart. Never invent prices. Be concise. End with 'Not financial advice.'"},
        {"role": "user", "content": f"Current view: {current_asset} on {current_tf}. User says: {user_msg}"}
    ]

    # -------- TIER 1: TRY GROQ --------
    if groq_key:
        try:
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "tools": TOOLS, "tool_choice": "auto", "temperature": 0.1
            }
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            
            # Handle tool calls
            if msg.get("tool_calls"):
                messages.append(msg)
                new_asset, new_tf = no_update, no_update
                for tool in msg["tool_calls"]:
                    args = json.loads(tool["function"]["arguments"])
                    sym, tf = args["asset"], args["timeframe"]
                    
                    if tool["function"]["name"] == "change_chart":
                        new_asset, new_tf = sym, tf
                        result = f"Chart changed to {sym} on {tf}."
                    else:
                        df = add_indicators(fetch_data(ASSETS[sym], tf))
                        last = df.iloc[-1]
                        result = f"{sym} | Price: {last['close']:.4f} | RSI: {last['rsi']:.2f}"
                    messages.append({"role": "tool", "tool_call_id": tool["id"], "content": result})
                
                payload["messages"] = messages
                payload.pop("tools")
                r2 = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
                return f"[Groq] {r2.json()['choices'][0]['message']['content']}", new_asset, new_tf
            else:
                return f"[Groq] {msg['content']}", no_update, no_update
        except Exception as e:
            print(f"Groq failed: {e}. Falling back to Pollinations...")

    # -------- TIER 2: TRY POLLINATIONS --------
    if poll_key:
        try:
            headers = {"Authorization": f"Bearer {poll_key}", "Content-Type": "application/json"}
            payload = {
                "model": "openai",
                "messages": messages,
                "tools": TOOLS, "tool_choice": "auto", "temperature": 0.1
            }
            r = requests.post("https://text.pollinations.ai/openai", headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            
            if msg.get("tool_calls"):
                messages.append(msg)
                new_asset, new_tf = no_update, no_update
                for tool in msg["tool_calls"]:
                    args = json.loads(tool["function"]["arguments"])
                    sym, tf = args["asset"], args["timeframe"]
                    if tool["function"]["name"] == "change_chart":
                        new_asset, new_tf = sym, tf
                        result = f"Chart changed to {sym} on {tf}."
                    else:
                        df = add_indicators(fetch_data(ASSETS[sym], tf))
                        last = df.iloc[-1]
                        result = f"{sym} | Price: {last['close']:.4f} | RSI: {last['rsi']:.2f}"
                    messages.append({"role": "tool", "tool_call_id": tool["id"], "content": result})
                
                payload["messages"] = messages
                payload.pop("tools")
                r2 = requests.post("https://text.pollinations.ai/openai", headers=headers, json=payload, timeout=30)
                return f"[Pollinations] {r2.json()['choices'][0]['message']['content']}", new_asset, new_tf
            else:
                return f"[Pollinations] {msg['content']}", no_update, no_update
        except Exception as e:
            return f"⚠️ Both AIs failed.\nGroq error/rate-limit.\nPollinations error: {e}", no_update, no_update

    return "⚠️ AI failed to respond.", no_update, no_update

# ==========================================
# 3. UI
# ==========================================
app = Dash(__name__, title="Pro AI Terminal")
app.layout = html.Div([
    html.Div([
        html.H2("📈 PRO AI TERMINAL", style={"margin": "0", "color": "#00d47e", "fontWeight": "bold"}),
        html.P("Groq + Pollinations Fallback · 5s Live Chart", style={"margin": "0", "color": "#888", "fontSize": "12px"})
    ], style={"padding": "15px 20px", "backgroundColor": "#131722", "borderBottom": "1px solid #2a2e39"}),

    html.Div([
        html.Div([
            html.Label("Asset", style={"color": "#aaa", "fontSize": "12px"}),
            dcc.Dropdown(id="asset-dropdown", options=[{"label": k, "value": k} for k in ASSETS.keys()], value="BTC/USDT", clearable=False, style={"color": "#000"})
        ], style={"flex": "1", "minWidth": "150px"}),
        html.Div([
            html.Label("Timeframe", style={"color": "#aaa", "fontSize": "12px"}),
            dcc.Dropdown(id="tf-dropdown", options=[{"label": k, "value": k} for k in ["5m", "15m", "1h", "4h", "1d"]], value="5m", clearable=False, style={"color": "#000"})
        ], style={"flex": "1", "minWidth": "100px"})
    ], style={"display": "flex", "gap": "15px", "padding": "15px 20px", "backgroundColor": "#1e222d"}),

    html.Div([
        dcc.Graph(id="main-chart", config={"displayModeBar": False}),
        dcc.Interval(id="refresh-timer", interval=5000)
    ], style={"borderBottom": "1px solid #2a2e39"}),

    html.Div([
        html.Div(id="chat-log", style={"flex": "1", "overflowY": "auto", "padding": "15px", "display": "flex", "flexDirection": "column", "gap": "10px", "minHeight": "200px", "maxHeight": "300px"}),
        html.Div([
            dcc.Input(id="chat-input", type="text", placeholder="E.g., 'Show ETH on 1h' or 'Analyze Gold'", n_submit=0,
                      style={"flex": "1", "padding": "12px", "borderRadius": "5px", "border": "1px solid #2a2e39", "backgroundColor": "#1e222d", "color": "#fff"}),
            html.Button("Send", id="chat-btn", n_clicks=0, style={"padding": "12px 20px", "backgroundColor": "#00d47e", "color": "#000", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontWeight": "bold"})
        ], style={"display": "flex", "gap": "10px", "padding": "15px", "backgroundColor": "#131722"})
    ], style={"backgroundColor": "#131722"})

], style={"backgroundColor": "#131722", "color": "#d1d4dc", "fontFamily": "system-ui, sans-serif", "height": "100vh", "display": "flex", "flexDirection": "column", "margin": "0"})

@app.callback(
    Output("main-chart", "figure"),
    Input("asset-dropdown", "value"),
    Input("tf-dropdown", "value"),
    Input("refresh-timer", "n_intervals")
)
def update_chart(asset, tf, _n):
    try:
        df = add_indicators(fetch_data(ASSETS[asset], tf))
        return build_chart(df, asset)
    except Exception as e:
        return go.Figure().update_layout(template="plotly_dark", title=f"Error: {e}")

@app.callback(
    Output("chat-log", "children"),
    Output("chat-input", "value"),
    Output("asset-dropdown", "value"),
    Output("tf-dropdown", "value"),
    Input("chat-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-log", "children"),
    State("asset-dropdown", "value"),
    State("tf-dropdown", "value"),
    prevent_initial_call=True
)
def chat(_n, _sub, text, log, current_asset, current_tf):
    if not text: return no_update, no_update, no_update, no_update
    log = list(log) if isinstance(log, list) else []

    log.append(html.Div(text, style={"alignSelf": "flex-end", "backgroundColor": "#2962ff", "color": "#fff", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px"}))

    reply_text, new_asset, new_tf = run_ai_agent(text, current_asset, current_tf)

    log.append(html.Div(reply_text, style={"alignSelf": "flex-start", "backgroundColor": "#2a2e39", "color": "#d1d4dc", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px", "whiteSpace": "pre-wrap"}))

    return log, "", new_asset, new_tf

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
