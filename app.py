import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="BTC Instant Predictor Plus", page_icon="⚡", layout="wide")
API_URL = "https://data-api.binance.vision/api/v3/klines"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"

st.title("⚡ Crypto Instant Trade Predictor Plus")
st.caption("Live market data · 5/10 minute outlook · Backtesting · Telegram alerts · Hindi")
st.warning("Educational tool only. Signals can be wrong; crypto is volatile. Do not use this app as the sole basis for real-money trades.")

with st.sidebar:
    st.header("Prediction settings")
    symbol = st.selectbox("Coin", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0)
    horizon_label = st.radio("Prediction horizon", ["5 minutes", "10 minutes"], index=0)
    horizon = 5 if horizon_label == "5 minutes" else 10
    st.caption("Public Binance Spot API. No exchange API key is needed.")

    st.divider()
    st.header("Telegram alerts")
    st.caption("Bot token/chat ID are used only when you press Send Telegram Alert. Do not put them in GitHub.")
    telegram_token = st.text_input("Telegram bot token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id = st.text_input("Telegram chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))
    st.caption("Create a bot with @BotFather, then start a chat with it. Find your chat ID using your bot's getUpdates endpoint.")

@st.cache_data(ttl=10, show_spinner=False)
def fetch_data(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    r = requests.get(
        API_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
        headers={"User-Agent": "CryptoInstantPredictorPlus/1.0"},
    )
    r.raise_for_status()
    raw = r.json()
    if not isinstance(raw, list) or not raw:
        raise ValueError("Public API se candle data nahi mila.")
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = ag / al.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    # Short-term momentum over 5 candles; used consistently for live and backtest.
    df["momentum5"] = (df["close"] / df["close"].shift(5) - 1) * 100
    score = (
        (df["ema9"] > df["ema21"]).astype(int) * 2 - 1
        + (df["close"] > df["ema9"]).astype(int) * 2 - 1
        + (df["momentum5"] > 0.04).astype(int)
        - (df["momentum5"] < -0.04).astype(int)
        + ((df["rsi"] >= 52) & (df["rsi"] < 68)).astype(int)
        - ((df["rsi"] > 32) & (df["rsi"] <= 48)).astype(int)
    )
    df["score"] = score
    df["signal"] = "WAIT"
    df.loc[df["score"] >= 3, "signal"] = "BUY"
    df.loc[df["score"] <= -3, "signal"] = "SELL"
    return df

def send_telegram(token: str, chat_id: str, message: str):
    if not token.strip() or not chat_id.strip():
        raise ValueError("Pehle Telegram bot token aur chat ID bharein.")
    url = TELEGRAM_URL.format(token=token.strip())
    response = requests.post(url, data={"chat_id": chat_id.strip(), "text": message}, timeout=15)
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise ValueError(str(body))
    return True

def evaluate_backtest(df: pd.DataFrame, horizon_mins: int):
    # Use only rows with enough indicator history and enough future candles.
    # Signal at candle i is compared to close i+horizon_mins. This is a simple
    # direction check and does not include fees, slippage, spread or execution.
    test = df.copy()
    test["future_close"] = test["close"].shift(-horizon_mins)
    test = test.dropna(subset=["future_close", "momentum5"]).copy()
    test = test[test["signal"].isin(["BUY", "SELL"])]
    if test.empty:
        return test, {"signals": 0, "wins": 0, "win_rate": None, "buy_n": 0, "buy_wins": 0, "sell_n": 0, "sell_wins": 0}
    test["actual_return_pct"] = (test["future_close"] / test["close"] - 1) * 100
    # Direction is counted correct only if future close moved in signal direction.
    test["win"] = ((test["signal"] == "BUY") & (test["actual_return_pct"] > 0)) | (
        (test["signal"] == "SELL") & (test["actual_return_pct"] < 0)
    )
    n = len(test)
    wins = int(test["win"].sum())
    buys = test[test["signal"] == "BUY"]
    sells = test[test["signal"] == "SELL"]
    return test, {
        "signals": n, "wins": wins, "win_rate": wins / n * 100 if n else None,
        "buy_n": len(buys), "buy_wins": int(buys["win"].sum()),
        "sell_n": len(sells), "sell_wins": int(sells["win"].sum())
    }

# Buttons are explicit user actions. Backtest and signal use current fetched data.
cbtn1, cbtn2 = st.columns(2)
with cbtn1:
    predict_clicked = st.button("⚡ PREDICT NOW", type="primary", use_container_width=True)
with cbtn2:
    refresh_clicked = st.button("🔄 Refresh market data", use_container_width=True)
if predict_clicked or refresh_clicked:
    st.cache_data.clear()

try:
    with st.spinner("Live candles load ho rahi hain..."):
        raw_df = fetch_data(symbol, "1m", 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        raw_df = raw_df[raw_df["close_time"] < now_ms].reset_index(drop=True)  # closed candles only
        data = add_indicators(raw_df)
        valid = data.dropna(subset=["momentum5"]).copy()
        if len(valid) < 50:
            raise ValueError("Abhi indicators/backtest ke liye paryapt candles nahi mili.")
        last = valid.iloc[-1]
except Exception as e:
    st.error("Live data load nahi ho saka. Internet/API availability check karke dobara try karein.")
    st.caption(f"Technical detail: {e}")
    st.stop()

price = float(last["close"])
rsi = float(last["rsi"])
momentum = float(last["momentum5"])
signal = str(last["signal"])
score = int(last["score"])
support = float(valid["low"].tail(20).min())
resistance = float(valid["high"].tail(20).max())
vol_avg = float(valid["volume"].tail(20).mean())
volume_ratio = float(last["volume"]) / vol_avg if vol_avg else 0.0

reasons = []
reasons.append("EMA 9 EMA 21 se upar hai." if last["ema9"] > last["ema21"] else "EMA 9 EMA 21 se neeche hai.")
reasons.append("Price EMA 9 ke upar hai." if price > float(last["ema9"]) else "Price EMA 9 ke neeche hai.")
reasons.append(f"Recent 5-minute momentum {momentum:+.3f}% hai.")
if rsi >= 68:
    reasons.append(f"RSI {rsi:.1f}: overbought zone ke paas/andar ho sakta hai.")
elif rsi <= 32:
    reasons.append(f"RSI {rsi:.1f}: oversold zone ke paas/andar ho sakta hai.")
elif rsi >= 52:
    reasons.append(f"RSI {rsi:.1f}: momentum relatively positive hai.")
elif rsi <= 48:
    reasons.append(f"RSI {rsi:.1f}: momentum relatively weak hai.")
else:
    reasons.append(f"RSI {rsi:.1f}: neutral zone mein hai.")

if signal == "BUY":
    title = "🟢 BUY BIAS"
    explanation = f"Indicators bullish bias dikhate hain; agle {horizon} minute mein price upar ja sakta hai, lekin yeh guarantee nahi."
elif signal == "SELL":
    title = "🔴 SELL BIAS"
    explanation = f"Indicators bearish bias dikhate hain; agle {horizon} minute mein price neeche ja sakta hai, lekin yeh guarantee nahi."
else:
    title = "🟡 WAIT / NO CLEAR SIGNAL"
    explanation = "Signals clear direction par agree nahi kar rahe. Is heuristic ke mutabik wait karna behtar hai."

st.caption(f"Coin: {symbol} · Last analyzed candle UTC: {last['time']} · Horizon: {horizon} minutes")
if signal == "BUY":
    st.success(f"## {title}")
elif signal == "SELL":
    st.error(f"## {title}")
else:
    st.warning(f"## {title}")
st.write(explanation)
st.caption("Signal score probability nahi hai. Yeh trained AI forecast nahi, transparent rule-based heuristic hai.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last closed price", f"${price:,.2f}")
m2.metric("RSI (14)", f"{rsi:.1f}")
m3.metric("5-minute momentum", f"{momentum:+.3f}%")
m4.metric("Volume / avg", f"{volume_ratio:.2f}×")

st.subheader("📈 Recent market chart")
view = valid.tail(120)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=view["time"], open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="Price"))
fig.add_trace(go.Scatter(x=view["time"], y=view["ema9"], name="EMA 9", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=view["time"], y=view["ema21"], name="EMA 21", line=dict(width=1.5)))
fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=5,r=5,t=25,b=5), legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Approx. price levels")
    st.metric("Support area", f"${support:,.2f}")
    st.metric("Resistance area", f"${resistance:,.2f}")
    st.caption("Last 20 closed candles ke low/high se nikle rough levels.")
with right:
    st.subheader("Signal ke reasons")
    for reason in reasons:
        st.write("• " + reason)
    st.write(f"**Rule score:** {score} (confidence percentage nahi)")
    st.write("**Risk:** High — 5–10 minute ke price moves bahut noisy hote hain.")

st.divider()
st.header("📨 Telegram alert")
st.write("Current signal Telegram par bhejne ke liye token aur chat ID sidebar mein daalo, phir button dabao. Is app mein automatic background monitoring nahi hai; alert tabhi jayega jab tum button dabao.")
alert_message = (
    f"{title}\nCoin: {symbol}\nHorizon: {horizon} minutes\n"
    f"Last closed price: ${price:,.2f}\nRSI: {rsi:.1f}\n"
    f"5m momentum: {momentum:+.3f}%\n"
    f"Support: ${support:,.2f}\nResistance: ${resistance:,.2f}\n"
    f"Reason: {explanation}\n\nEducational signal only; not financial advice."
)
if st.button("📨 Send Telegram Alert", use_container_width=True):
    try:
        send_telegram(telegram_token, telegram_chat_id, alert_message)
        st.success("Telegram message bhej diya gaya.")
    except Exception as e:
        st.error("Telegram alert nahi bheja ja saka. Bot token, chat ID aur bot chat check karein.")
        st.caption(str(e))

st.divider()
st.header("🧪 Historical backtesting")
st.write(f"Is test mein historical 1-minute candles par wahi signal rules lagte hain aur check hota hai ki signal ke {horizon} minute baad close signal ki direction mein gaya ya nahi.")
st.caption("Yeh simplified directional hit rate hai—profitability nahi. Fees, slippage, spread, order fills aur overlapping signals include nahi hain. Historical performance future result guarantee nahi karta.")
test, stats = evaluate_backtest(valid, horizon)
if stats["signals"] == 0:
    st.info("Is sample mein koi BUY/SELL signal nahi mila. Zyada data ya baad mein dobara try karein.")
else:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Signals tested", f"{stats['signals']}")
    b2.metric("Directionally correct", f"{stats['wins']}")
    b3.metric("Historical hit rate", f"{stats['win_rate']:.1f}%")
    b4.metric("Horizon", f"{horizon} min")
    buy_rate = stats["buy_wins"] / stats["buy_n"] * 100 if stats["buy_n"] else None
    sell_rate = stats["sell_wins"] / stats["sell_n"] * 100 if stats["sell_n"] else None
    st.write(f"**BUY signals:** {stats['buy_wins']}/{stats['buy_n']} correct" + (f" ({buy_rate:.1f}%)" if buy_rate is not None else ""))
    st.write(f"**SELL signals:** {stats['sell_wins']}/{stats['sell_n']} correct" + (f" ({sell_rate:.1f}%)" if sell_rate is not None else ""))
    st.dataframe(
        test[["time", "signal", "close", "future_close", "actual_return_pct", "win"]]
        .tail(100).rename(columns={
            "time": "Signal time (UTC)", "signal": "Signal", "close": "Price at signal",
            "future_close": f"Price after {horizon}m", "actual_return_pct": "Return (%)", "win": "Direction correct"
        }),
        use_container_width=True, hide_index=True
    )
    csv = test.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download backtest CSV", data=csv, file_name=f"{symbol}_{horizon}m_backtest.csv", mime="text/csv")

st.info("Free API limits: public Binance data can be rate-limited or unavailable. Backtest is limited to the candles fetched in this run (up to 1,000 one-minute candles). For meaningful testing, use longer history and include fees/slippage.")
st.warning("Real-money use se pehle independent backtesting aur paper trading karo. App orders place nahi karta; exchange keys ya wallet seed phrase kabhi mat do.")
