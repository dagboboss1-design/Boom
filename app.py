import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update, ctx
from openai import OpenAI

# ==========================================
# 1. DATA & ASSETS
# ==========================================
ASSETS = {
    "BTC/USDT": "BTC/USDT", "ETH/USDT": "ETH/USDT", "SOL/USDT": "SOL/USDT",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "EUR/USD": "EURUSD=X", 
    "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "GBP/JPY": "GBPJPY=X", "DXY": "DX-Y.NYB"
}

def fetch_crypto_live(symbol, timeframe):
    for ex_name in ["binance", "kraken", "coinbase"]:
        try:
            ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("ts")
        except Exception:
            continue
    raise ValueError(f"Could not fetch live crypto for {symbol}")

def fetch_yahoo(symbol, timeframe):
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

def fetch_data(symbol, timeframe):
    if "/" in symbol: return fetch_crypto_live(symbol, timeframe)
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

# ==========================================
# 3. PLOTLY CHART
# ==========================================
def build_chart(df, symbol, selected_indicators):
    has_rsi = "RSI" in selected_indicators
    has_macd = "MACD" in selected_indicators
    rows = 1 + int(has_rsi) + int(has_macd)
    heights = [0.65] if rows == 1 else ([0.75, 0.25] if rows == 2 else [0.60, 0.20, 0.20])
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#00d47e", decreasing_line_color="#ff4d4d"), row=1, col=1)
    
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
        current_row += 1
    if has_macd:
        fig.add_trace(go.Bar(x=df.index, y=df["macd"]-df["macd_sig"], name="Hist", marker_color="#888"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#4da3ff", width=1)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_sig"], name="Signal", line=dict(color="#ffaa33", width=1)), row=current_row, col=1)

    fig.update_layout(template="plotly_dark", height=600 if rows == 1 else 750, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, uirevision="keep", paper_bgcolor="#131722", plot_bgcolor="#131722",
        legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(gridcolor="#2a2e39", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2e39", zeroline=False)
    return fig

# ==========================================
# 4. AGENTIC AI (TOOLS)
# ==========================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "change_chart",
            "description": "Change the asset and timeframe displayed on the user's chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": list(ASSETS.keys()), "description": "The exact asset symbol from the available list."},
                    "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"], "description": "The timeframe to view."}
                },
                "required": ["asset", "timeframe"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market",
            "description": "Fetch live data and indicators for a specific asset and timeframe to analyze the market.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": list(ASSETS.keys())},
                    "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]}
                },
                "required": ["asset", "timeframe"]
            }
        }
    }
]

def run_ai_agent(user_message, current_asset, current_tf, api_key):
    if not api_key: return "⚠️ OPENAI_API_KEY missing.", no_update, no_update
    
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a professional trading AI. You can control the user's chart using the 'change_chart' tool. You can get live data using 'analyze_market'. Never invent prices. Be concise. End with 'Not financial advice.'"},
        {"role": "user", "content": f"Current view: {current_asset} on {current_tf}. Question: {user_message}"}
    ]
    
    # 1. Ask OpenAI what to do
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages, tools=TOOLS, tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    new_asset = no_update
    new_tf = no_update

    # 2. Handle Tool Calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            
            if tool_call.function.name == "change_chart":
                new_asset = args["asset"]
                new_tf = args["timeframe"]
                tool_result = f"Successfully changed chart to {new_asset} on {new_tf}."
                
            elif tool_call.function.name == "analyze_market":
                df = add_indicators(fetch_data(ASSETS[args["asset"]], args["timeframe"]))
                last = df.iloc[-1]
                tool_result = f"Asset: {args['asset']}, Price: {last['close']:.4f}, RSI: {last['rsi']:.2f}, MACD: {last['macd']:.4f}"
            
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
        
        # 3. Get final response after tools ran
        second_response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages
        )
        final_text = second_response.choices[0].message.content
    else:
        final_text = response_message.content
        
    return final_text, new_asset, new_tf

# ==========================================
# 5. UI & DASHBOARD
# ==========================================
app = Dash(__name__, title="Pro Terminal")
app.layout = html.Div([
    html.Div([
        html.H2("📈 PRO TERMINAL", style={"margin": "0", "color": "#00d47e", "fontWeight": "bold"}),
        html.P("Live Market · Agentic AI", style={"margin": "0", "color": "#888", "fontSize": "12px"})
    ], style={"padding": "15px 20px", "backgroundColor": "#131722", "borderBottom": "1px solid #2a2e39"}),
    
    html.Div([
        html.Div([
            html.Label("Asset", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(id="asset-dropdown", options=[{"label": k, "value": k} for k in ASSETS.keys()], value="BTC/USDT", clearable=False, style={"color": "#000"})
        ], style={"flex": "1", "minWidth": "150px"}),
        
        html.Div([
            html.Label("Timeframe", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(id="tf-dropdown", options=[{"label": k, "value": k} for k in ["1m", "5m", "15m", "1h", "4h", "1d"]], value="5m", clearable=False, style={"color": "#000"})
        ], style={"flex": "1", "minWidth": "100px"}),
        
        html.Div([
            html.Label("Indicators", style={"color": "#aaa", "fontSize": "12px", "marginBottom": "5px"}),
            dcc.Dropdown(id="indicator-dropdown", options=[{"label": i, "value": i} for i in ["EMA 20", "EMA 50", "Bollinger Bands", "RSI", "MACD"]], value=["EMA 20", "EMA 50", "RSI"], multi=True, style={"color": "#000"})
        ], style={"flex": "2", "minWidth": "250px"})
    ], style={"display": "flex", "gap": "15px", "padding": "15px 20px", "backgroundColor": "#1e222d", "flexWrap": "wrap"}),

    html.Div([
        html.Div([dcc.Graph(id="main-chart", config={"displayModeBar": False}), dcc.Interval(id="refresh-timer", interval=5000)], style={"flex": "2", "minWidth": "300px"}),
        
        html.Div([
            html.Div("🤖 Agentic AI", style={"padding": "10px", "borderBottom": "1px solid #2a2e39", "fontWeight": "bold", "color": "#00d47e"}),
            html.Div(id="chat-log", style={"flex": "1", "overflowY": "auto", "padding": "15px", "display": "flex", "flexDirection": "column", "gap": "10px"}),
            html.Div([
                dcc.Input(id="chat-input", type="text", placeholder="Try: 'Show ETH on 1h' or 'Analyze Gold'", n_submit=0,
                          style={"flex": "1", "padding": "10px", "borderRadius": "5px", "border": "1px solid #2a2e39", "backgroundColor": "#1e222d", "color": "#fff"}),
                html.Button("Send", id="chat-btn", n_clicks=0, style={"padding": "10px 15px", "backgroundColor": "#00d47e", "color": "#000", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontWeight": "bold"})
            ], style={"display": "flex", "gap": "10px", "padding": "10px", "borderTop": "1px solid #2a2e39", "backgroundColor": "#131722"})
        ], style={"flex": "1", "minWidth": "300px", "backgroundColor": "#131722", "display": "flex", "flexDirection": "column", "borderLeft": "1px solid #2a2e39"})
    ], style={"display": "flex", "flexWrap": "wrap", "height": "calc(100vh - 130px)"})
], style={"backgroundColor": "#131722", "color": "#d1d4dc", "fontFamily": "system-ui, sans-serif", "minHeight": "100vh"})

@app.callback(
    Output("main-chart", "figure"),
    Input("asset-dropdown", "value"),
    Input("tf-dropdown", "value"),
    Input("indicator-dropdown", "value"),
    Input("refresh-timer", "n_intervals")
)
def update_chart(asset, tf, indicators, _n):
    try:
        df = add_indicators(fetch_data(ASSETS[asset], tf))
        return build_chart(df, asset, indicators or [])
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
    
    # 1. Show user message
    log.append(html.Div(text, style={"alignSelf": "flex-end", "backgroundColor": "#2962ff", "color": "#fff", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px"}))
    
    # 2. Run AI Agent
    try:
        reply, new_asset, new_tf = run_ai_agent(text, current_asset, current_tf, os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        reply, new_asset, new_tf = f"Error: {e}", no_update, no_update
        
    # 3. Show AI message
    log.append(html.Div(reply, style={"alignSelf": "flex-start", "backgroundColor": "#2a2e39", "color": "#d1d4dc", "padding": "8px 12px", "borderRadius": "10px", "maxWidth": "85%", "fontSize": "13px", "whiteSpace": "pre-wrap"}))
    
    return log, "", new_asset, new_tf

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
