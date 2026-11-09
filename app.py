"""Live multi-market chart + AI chat. No auth. Ready for Railway."""
import os
from datetime import datetime
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update


# ============ 1. DATA ============
def fetch_crypto(symbol, timeframe, limit=200):
    for name in ["binance", "kraken", "coinbase"]:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("ts")
        except Exception:
            continue
    raise RuntimeError(f"No exchange could fetch {symbol}")


TF_MAP = {"1m":("7d","1m"), "5m":("30d","5m"), "15m":("60d","15m"),
          "1h":("90d","1h"), "4h":("90d","1h"), "1d":("2y","1d")}


def fetch_yf(symbol, timeframe):
    period, interval = TF_MAP.get(timeframe, ("90d","1h"))
    raw = yf.download(symbol, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open","high","low","close","volume"]]
    if timeframe == "4h":
        df = df.resample("4h").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna()
    return df.dropna()


def fetch_data(symbol, timeframe):
    if "/" in symbol:
        return fetch_crypto(symbol, timeframe)
    return fetch_yf(symbol, timeframe)


# ============ 2. MATH ============
def rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def enrich(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df["close"])
    ef = df["close"].ewm(span=12, adjust=False).mean()
    es = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ef - es
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = sma + 2*std
    df["bb_lower"] = sma - 2*std
    df["pctb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df.dropna()


def signal(df):
    if len(df) < 60:
        return {"signal": "INSUFFICIENT DATA", "votes": "—", "notes": []}
    last = df.iloc[-1]
    bull = bear = 0
    notes = []
    if last["ema20"] > last["ema50"]:
        bull += 1; notes.append("EMA20>EMA50")
    else:
        bear += 1; notes.append("EMA20<EMA50")
    r = last["rsi"]
    if 50 < r < 70: bull += 1; notes.append(f"RSI {r:.1f} bull")
    elif 30 < r < 50: bear += 1; notes.append(f"RSI {r:.1f} bear")
    if last["macd"] > last["macd_sig"]:
        bull += 1; notes.append("MACD>sig")
    else:
        bear += 1; notes.append("MACD<sig")
    b = last["pctb"]
    if 0.5 < b < 0.95: bull += 1; notes.append(f"%B {b:.2f}")
    elif 0.05 < b < 0.5: bear += 1; notes.append(f"%B {b:.2f}")
    if bull >= 3 and bull > bear: sig = "BUY"
    elif bear >= 3 and bear > bull: sig = "SELL"
    else: sig = "HOLD"
    return {
        "signal": sig, "votes": f"{bull}B/{bear}S", "notes": notes,
        "price": round(float(last["close"]), 6),
        "rsi": round(float(r), 2),
        "atr": round(float(last["atr"]), 6),
    }


# ============ 3. CHART ============
def build_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.60, 0.20, 0.20],
                        vertical_spacing=0.03,
                        subplot_titles=("", "Volume", "RSI"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="#00d47e",
        decreasing_line_color="#ff4d4d",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20",
                             line=dict(color="#4da3ff", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50",
                             line=dict(color="#ffaa33", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Up",
                             line=dict(color="#666", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Low",
                             line=dict(color="#666", width=1, dash="dot"),
                             fill="tonexty",
                             fillcolor="rgba(120,120,120,0.05)"), row=1, col=1)
    colors = ["#00d47e" if c >= o else "#ff4d4d"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"],
                         marker_color=colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                             line=dict(color="#f59e0b", width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#555", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#555", row=3, col=1)
    fig.update_layout(
        template="plotly_dark", height=750, title=symbol,
        showlegend=True, xaxis_rangeslider_visible=False,
        uirevision="keep",
        margin=dict(l=40, r=20, t=50, b=30),
    )
    fig.update_yaxes(range=[0,100], row=3, col=1)
    return fig


# ============ 4. AI ============
def offline_answer(message, symbol, timeframe):
    return (
        f"Offline mode — no OPENAI_API_KEY configured.\n\n"
        f"Symbol: {symbol} · TF: {timeframe}\n\n"
        f"Ask about RSI, MACD, Bollinger, trend, or risk. "
        f"To enable live AI, set OPENAI_API_KEY in Railway.\n\n"
        f"Analysis, not financial advice."
    )


def chat_with_ai(message, symbol, timeframe, ctx, api_key):
    if not api_key:
        return offline_answer(message, symbol, timeframe)
    try:
        import httpx
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content":
                     f"You are a trading analyst. Live context: {ctx}. "
                     f"Never invent prices. Be concise. End with "
                     f"'Analysis, not financial advice.'"},
                    {"role": "user", "content": message},
                ],
                "max_tokens": 500,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[AI error: {e}]\n\n" + offline_answer(message, symbol, timeframe)


# ============ 5. UI ============
WATCHLIST = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT",
    "AAPL","NVDA","SPY","TSLA",
    "EURUSD=X","GBPUSD=X","USDJPY=X",
]

app = Dash(__name__, title="Live Trader")
app.layout = html.Div([
    html.Div([
        html.H1("📈 Live Trader", style={"margin":"0","color":"#00d47e"}),
        html.P("Real-time charts · AI analyst · No signup",
               style={"margin":"4px 0","color":"#888","fontSize":"13px"}),
    ], style={"padding":"20px","borderBottom":"1px solid #222"}),

    dcc.Tabs(id="tabs", value="chart", children=[
        dcc.Tab(label="📊 Chart", value="chart", children=[
            html.Div([
                dcc.Dropdown(id="symbol",
                    options=[{"label":s,"value":s} for s in WATCHLIST],
                    value="BTC/USDT", clearable=False,
                    style={"width":"180px"}),
                dcc.Dropdown(id="tf",
                    options=[{"label":t,"value":t} for t in
                             ["1m","5m","15m","1h","4h","1d"]],
                    value="5m", clearable=False, style={"width":"100px"}),
                dcc.Checklist(id="auto",
                    options=[{"label":" Auto-refresh","value":"on"}],
                    value=["on"],
                    style={"color":"#eee","fontSize":"13px"}),
            ], style={"display":"flex","gap":"12px","padding":"15px",
                      "alignItems":"center","flexWrap":"wrap"}),
            html.Div(id="signal-box",
                style={"padding":"12px 20px","fontFamily":"monospace",
                       "fontSize":"14px","color":"#ccc"}),
            dcc.Graph(id="chart", style={"height":"750px"}),
            dcc.Interval(id="timer", interval=5000),
        ]),
        dcc.Tab(label="🤖 AI Chat", value="ai", children=[
            html.Div(id="chat-log",
                style={"padding":"20px","height":"calc(100vh - 320px)",
                       "overflowY":"auto","display":"flex",
                       "flexDirection":"column","gap":"10px"}),
            html.Div([
                dcc.Input(id="chat-input", type="text",
                    placeholder="e.g. analyze BTC/USDT on 5m",
                    n_submit=0,
                    style={"flex":"1","padding":"12px","borderRadius":"8px",
                           "border":"1px solid #444","backgroundColor":"#1a1a1a",
                           "color":"#eee","fontSize":"14px"}),
                html.Button("Send", id="send-btn", n_clicks=0,
                    style={"padding":"12px 24px","borderRadius":"8px",
                           "border":"none","backgroundColor":"#00d47e",
                           "color":"#000","fontWeight":"bold",
                           "cursor":"pointer"}),
            ], style={"display":"flex","gap":"8px","padding":"15px",
                      "borderTop":"1px solid #222"}),
        ]),
    ], style={"padding":"0 20px"}),
], style={"backgroundColor":"#111","color":"#eee",
          "fontFamily":"system-ui,sans-serif","minHeight":"100vh"})


@app.callback(Output("timer","disabled"), Input("auto","value"))
def toggle_auto(auto):
    return "on" not in (auto or [])


@app.callback(
    Output("chart","figure"),
    Output("signal-box","children"),
    Input("symbol","value"),
    Input("tf","value"),
    Input("timer","n_intervals"),
)
def update_chart(symbol, tf, _n):
    try:
        df = enrich(fetch_data(symbol, tf)).tail(200)
        sig = signal(df)
        color = {"BUY":"#00d47e","SELL":"#ff4d4d",
                 "HOLD":"#ffcc00"}.get(sig["signal"],"#888")
        box = html.Div([
            html.Span(f"{symbol}  ", style={"fontWeight":"bold"}),
            html.Span(f"{sig.get('price','—')}  ", style={"fontSize":"16px"}),
            html.Span(f"SIGNAL: {sig['signal']}  ",
                      style={"color":color,"fontWeight":"bold"}),
            html.Span(f"({sig['votes']})   RSI {sig.get('rsi','—')}   "),
            html.Span(" · ".join(sig.get("notes",[])[:3]),
                      style={"color":"#888","fontSize":"12px"}),
        ])
        return build_chart(df, symbol), box
    except Exception as e:
        empty = go.Figure().update_layout(
            template="plotly_dark", height=750,
            title=f"⚠ {symbol} — {e}")
        return empty, html.Div(f"⚠ {e}", style={"color":"#ff4d4d"})


@app.callback(
    Output("chat-log","children"),
    Output("chat-input","value"),
    Input("send-btn","n_clicks"),
    Input("chat-input","n_submit"),
    State("chat-input","value"),
    State("chat-log","children"),
    State("symbol","value"),
    State("tf","value"),
    prevent_initial_call=True,
)
def on_chat(_n, _sub, text, log, symbol, tf):
    if not text or not text.strip():
        return no_update, no_update
    log = list(log) if isinstance(log, list) else []
    log.append(html.Div(text, style={
        "alignSelf":"flex-end","backgroundColor":"#2563eb","color":"#fff",
        "padding":"10px 14px","borderRadius":"14px 14px 2px 14px",
        "maxWidth":"75%","fontSize":"14px"}))
    ctx = "no context"
    try:
        df = enrich(fetch_data(symbol, tf)).tail(100)
        s = signal(df)
        ctx = f"symbol={symbol}, tf={tf}, price={s.get('price')}, " \
              f"signal={s['signal']}, votes={s['votes']}, rsi={s.get('rsi')}"
    except Exception:
        pass
    reply = chat_with_ai(text, symbol, tf, ctx,
                         os.getenv("OPENAI_API_KEY",""))
    log.append(html.Div(reply, style={
        "alignSelf":"flex-start","backgroundColor":"#1a1a1a","color":"#eee",
        "padding":"10px 14px","borderRadius":"14px 14px 14px 2px",
        "maxWidth":"85%","fontSize":"14px","whiteSpace":"pre-wrap"}))
    return log, ""


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    print(f"→ http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
