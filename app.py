import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Crypto Instant Predictor Plus", page_icon="⚡", layout="wide")

BINANCE_API = "https://data-api.binance.vision/api/v3/klines"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
INDIA_TZ = ZoneInfo("Asia/Kolkata")

st.title("⚡ Crypto Instant Trade Predictor Plus")
st.caption("Live market data · Multiple horizons · Technical indicators · Hindi")
st.warning(
    "Educational tool only. Signals can be wrong; crypto is volatile. "
    "Do not use this app as the sole basis for real-money trades."
)

COINS = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Solana (SOL)": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "TRON (TRX)": "TRXUSDT",
    "Avalanche (AVAX)": "AVAXUSDT",
    "Chainlink (LINK)": "LINKUSDT",
}
EXCHANGES = ["Binance", "CoinSwitch", "CoinDCX", "WazirX"]
HORIZONS = [1, 2, 5, 10, 15, 30, 60]

with st.sidebar:
    st.header("Prediction settings")
    coin_name = st.selectbox("Coin Call / Coin select", list(COINS.keys()))
    symbol = COINS[coin_name]
    exchange = st.selectbox("Exchange", EXCHANGES, index=0)
    horizon = st.selectbox(
        "Prediction horizon",
        HORIZONS,
        index=2,
        format_func=lambda n: f"{n} minute" if n == 1 else f"{n} minutes",
    )
    st.caption(
        "Live candles is version mein Binance public Spot API se aate hain. "
        "CoinSwitch, CoinDCX aur WazirX options diye gaye hain, lekin unke "
        "alag live API adapters abhi connect nahi hain."
    )
    st.divider()
    st.header("Date & time")
    st.caption("Prediction start ke liye phone/browser ka current local time use karein.")
    # Streamlit server code cannot directly read a phone's clock. This uses India
    # local time, which matches phones configured to India Standard Time.
    now_local = datetime.now(INDIA_TZ)
    selected_date = st.date_input("Prediction date", value=now_local.date())
    selected_time = st.time_input("Prediction start time", value=now_local.time().replace(microsecond=0))
    use_current_time = st.checkbox("Use current India/mobile local time", value=True)
    if use_current_time:
        start_time = datetime.now(INDIA_TZ)
    else:
        start_time = datetime.combine(selected_date, selected_time).replace(tzinfo=INDIA_TZ)
    expected_end = start_time + timedelta(minutes=int(horizon))
    st.caption(f"Start: {start_time.strftime('%d-%m-%Y %I:%M:%S %p IST')}")
    st.caption(f"Target time: {expected_end.strftime('%d-%m-%Y %I:%M:%S %p IST')}")
    st.divider()
    st.header("Telegram alerts")
    st.caption("Token/chat ID sirf alert bhejte waqt use hote hain. Inhe GitHub par upload na karein.")
    telegram_token = st.text_input("Telegram bot token", type="password", value=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id = st.text_input("Telegram chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))


@st.cache_data(ttl=10, show_spinner=False)
def fetch_data(symbol: str, interval: str = "1m", limit: int = 1000) -> pd.DataFrame:
    response = requests.get(
        BINANCE_API,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
        headers={"User-Agent": "CryptoInstantPredictorPlus/2.0"},
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or not raw:
        raise ValueError("Public API se candle data nahi mila.")
    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    for col in ["open", "high", "low", "close", "volume", "close_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    return df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    df["momentum5"] = (df["close"] / df["close"].shift(5) - 1) * 100
    score = (
        ((df["ema9"] > df["ema21"]).astype(int) * 2 - 1)
        + ((df["close"] > df["ema9"]).astype(int) * 2 - 1)
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


def send_telegram(token: str, chat_id: str, message: str) -> None:
    if not token.strip() or not chat_id.strip():
        raise ValueError("Pehle Telegram bot token aur chat ID bharein.")
    response = requests.post(
        TELEGRAM_URL.format(token=token.strip()),
        data={"chat_id": chat_id.strip(), "text": message},
        timeout=15,
    )
    response.raise_for_status()
    if not response.json().get("ok"):
        raise ValueError("Telegram API ne message accept nahi kiya.")


def evaluate_backtest(df: pd.DataFrame, horizon_mins: int):
    test = df.copy()
    test["future_close"] = test["close"].shift(-horizon_mins)
    test = test.dropna(subset=["future_close", "momentum5"]).copy()
    test = test[test["signal"].isin(["BUY", "SELL"])]
    if test.empty:
        return test, {"signals": 0, "wins": 0, "win_rate": None, "buy_n": 0, "buy_wins": 0, "sell_n": 0, "sell_wins": 0}
    test["actual_return_pct"] = (test["future_close"] / test["close"] - 1) * 100
    test["win"] = (
        ((test["signal"] == "BUY") & (test["actual_return_pct"] > 0))
        | ((test["signal"] == "SELL") & (test["actual_return_pct"] < 0))
    )
    buys = test[test["signal"] == "BUY"]
    sells = test[test["signal"] == "SELL"]
    return test, {
        "signals": len(test),
        "wins": int(test["win"].sum()),
        "win_rate": float(test["win"].mean() * 100),
        "buy_n": len(buys),
        "buy_wins": int(buys["win"].sum()),
        "sell_n": len(sells),
        "sell_wins": int(sells["win"].sum()),
    }


c1, c2 = st.columns(2)
with c1:
    predict_clicked = st.button("⚡ PREDICT NOW", type="primary", use_container_width=True)
with c2:
    refresh_clicked = st.button("🔄 Refresh market data", use_container_width=True)
if predict_clicked or refresh_clicked:
    st.cache_data.clear()

if exchange != "Binance":
    st.info(
        f"{exchange} select kiya hai. Filhaal live price data Binance public API se liya ja raha hai; "
        "isliye signal Binance Spot market ka hai, selected exchange ka confirmed signal nahi. "
        "Exchange-specific live integration alag se karni hogi."
    )

try:
    with st.spinner("Live market candles load ho rahi hain..."):
        raw_df = fetch_data(symbol, "1m", 1000)
        now_ms = int(datetime.now(ZoneInfo("UTC")).timestamp() * 1000)
        raw_df = raw_df[raw_df["close_time"] < now_ms].reset_index(drop=True)
        data = add_indicators(raw_df)
        valid = data.dropna(subset=["momentum5"]).copy()
        if len(valid) < 50:
            raise ValueError("Indicators ke liye paryapt closed candles nahi mili.")
        last = valid.iloc[-1]
except Exception as exc:
    st.error("Live data load nahi ho saka. Internet/API availability check karke dobara try karein.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

price = float(last["close"])
rsi = float(last["rsi"])
momentum = float(last["momentum5"])
signal = str(last["signal"])
score = int(last["score"])
support = float(valid["low"].tail(20).min())
resistance = float(valid["high"].tail(20).max())
avg_volume = float(valid["volume"].tail(20).mean())
volume_ratio = float(last["volume"]) / avg_volume if avg_volume else 0.0

reasons = [
    "EMA 9, EMA 21 se upar hai." if last["ema9"] > last["ema21"] else "EMA 9, EMA 21 se neeche hai.",
    "Price EMA 9 ke upar hai." if price > float(last["ema9"]) else "Price EMA 9 ke neeche hai.",
    f"Recent 5-minute momentum {momentum:+.3f}% hai.",
]
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
    explanation = f"Indicators bullish bias dikhate hain; agle {horizon} minute mein price upar ja sakta hai, lekin guarantee nahi."
elif signal == "SELL":
    title = "🔴 SELL BIAS"
    explanation = f"Indicators bearish bias dikhate hain; agle {horizon} minute mein price neeche ja sakta hai, lekin guarantee nahi."
else:
    title = "🟡 WAIT / NO CLEAR SIGNAL"
    explanation = "Indicators clear direction par agree nahi kar rahe; is rule-based method ke mutabik wait karna behtar hai."

st.subheader("Prediction summary")
st.caption(
    f"Coin: {coin_name} ({symbol}) · Data source: Binance Spot · "
    f"Last analyzed candle (UTC): {last['time']} · Horizon: {horizon} min"
)
st.caption(
    f"Prediction start: {start_time.strftime('%d-%m-%Y %I:%M:%S %p %Z')} · "
    f"Target time: {expected_end.strftime('%d-%m-%Y %I:%M:%S %p %Z')}"
)
if signal == "BUY":
    st.success(f"## {title}")
elif signal == "SELL":
    st.error(f"## {title}")
else:
    st.warning(f"## {title}")
st.write(explanation)
st.caption("Signal score probability nahi hai. Yeh trained AI forecast nahi, rule-based indicator hai.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last closed price", f"${price:,.6f}")
m2.metric("RSI (14)", f"{rsi:.1f}")
m3.metric("5-minute momentum", f"{momentum:+.3f}%")
m4.metric("Volume / average", f"{volume_ratio:.2f}×")

st.subheader("📈 Recent market chart")
view = valid.tail(120)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=view["time"], open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="Price"))
fig.add_trace(go.Scatter(x=view["time"], y=view["ema9"], name="EMA 9", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=view["time"], y=view["ema21"], name="EMA 21", line=dict(width=1.5)))
fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=25, b=5), legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Approx. price levels")
    st.metric("Support area", f"${support:,.6f}")
    st.metric("Resistance area", f"${resistance:,.6f}")
    st.caption("Last 20 closed candles ke low/high se nikle rough levels.")
with right:
    st.subheader("Signal ke reasons")
    for reason in reasons:
        st.write("• " + reason)
    st.write(f"**Rule score:** {score} (confidence percentage nahi)")
    st.write(f"**Risk:** High — {horizon}-minute moves noisy ho sakte hain.")

st.divider()
st.header("📨 Telegram alert")
st.write("Signal Telegram par bhejne ke liye token aur chat ID sidebar mein bharein, phir button dabayein. Automatic background monitoring nahi hai.")
alert_message = (
    f"{title}\nCoin: {symbol}\nExchange selected: {exchange}\n"
    f"Data source: Binance Spot\nHorizon: {horizon} minutes\n"
    f"Prediction start: {start_time.isoformat()}\nTarget time: {expected_end.isoformat()}\n"
    f"Last closed price: ${price:,.6f}\nRSI: {rsi:.1f}\n"
    f"5m momentum: {momentum:+.3f}%\nSupport: ${support:,.6f}\n"
    f"Resistance: ${resistance:,.6f}\nReason: {explanation}\n\n"
    "Educational signal only; not financial advice."
)
if st.button("📨 Send Telegram Alert", use_container_width=True):
    try:
        send_telegram(telegram_token, telegram_chat_id, alert_message)
        st.success("Telegram message bhej diya gaya.")
    except Exception as exc:
        st.error("Telegram alert nahi bheja ja saka. Bot token, chat ID aur bot chat check karein.")
        st.caption(str(exc))

st.divider()
st.header("🧪 Historical backtesting")
st.write(f"Historical 1-minute candles par signal rules laga kar check hota hai ki signal ke {horizon} minute baad close signal ki direction mein gaya ya nahi.")
st.caption("Directional hit rate profitability nahi hai. Fees, slippage, spread, order fills aur overlapping signals include nahi hain.")
test, stats = evaluate_backtest(valid, int(horizon))
if stats["signals"] == 0:
    st.info("Is sample mein koi BUY/SELL signal nahi mila. Baad mein dobara try karein.")
else:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Signals tested", str(stats["signals"]))
    b2.metric("Directionally correct", str(stats["wins"]))
    b3.metric("Historical hit rate", f"{stats['win_rate']:.1f}%")
    b4.metric("Horizon", f"{horizon} min")
    buy_rate = stats["buy_wins"] / stats["buy_n"] * 100 if stats["buy_n"] else None
    sell_rate = stats["sell_wins"] / stats["sell_n"] * 100 if stats["sell_n"] else None
    st.write(f"**BUY signals:** {stats['buy_wins']}/{stats['buy_n']} correct" + (f" ({buy_rate:.1f}%)" if buy_rate is not None else ""))
    st.write(f"**SELL signals:** {stats['sell_wins']}/{stats['sell_n']} correct" + (f" ({sell_rate:.1f}%)" if sell_rate is not None else ""))
    st.dataframe(
        test[["time", "signal", "close", "future_close", "actual_return_pct", "win"]].tail(100).rename(
            columns={
                "time": "Signal time (UTC)",
                "signal": "Signal",
                "close": "Price at signal",
                "future_close": f"Price after {horizon}m",
                "actual_return_pct": "Return (%)",
                "win": "Direction correct",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download backtest CSV",
        data=test.to_csv(index=False).encode("utf-8"),
        file_name=f"{symbol}_{horizon}m_backtest.csv",
        mime="text/csv",
    )

st.info("Public Binance API rate-limited ya unavailable ho sakti hai. Backtest is run mein fetch hui candles (maximum 1,000 one-minute candles) tak limited hai.")
st.warning("Real-money use se pehle independent backtesting aur paper trading karein. App orders place nahi karta. Exchange API keys ya wallet seed phrase kabhi share na karein.")
