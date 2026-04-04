import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
import os
import uuid
import streamlit.components.v1 as components
from datetime import datetime, time as dt_time, timedelta
from FinMind.data import DataLoader
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ===========================================================================
# --- 0. 基礎設定 ---
# ===========================================================================
tw_tz        = pytz.timezone("Asia/Taipei")
MARKET_OPEN  = dt_time(9, 0)
MARKET_CLOSE = dt_time(13, 30)
TG_SAVE_FILE = "tg_config.json"
USER_DATA_DIR = "user_data"          # 每個使用者的股票清單存放目錄
LS_KEY        = "tw_stock_browser_id"  # localStorage key，存使用者唯一 ID
DEFAULT_STOCKS = [{"id": "2330", "name": "台積電"}]

os.makedirs(USER_DATA_DIR, exist_ok=True)


def now_tw() -> datetime:
    return datetime.now(tw_tz)


def is_market_open() -> bool:
    n = now_tw()
    if n.weekday() >= 5:
        return False
    return MARKET_OPEN <= n.time() <= MARKET_CLOSE


def today_str() -> str:
    return now_tw().strftime("%Y-%m-%d")


# ===========================================================================
# --- 1. 使用者識別：browser_id（存在 localStorage）---
# ===========================================================================

def get_browser_id_component():
    """
    注入 JS：
    - 讀取 localStorage 的 browser_id（沒有就建立一個新的 UUID）
    - 把 browser_id 寫入 URL query param ?bid=xxx，讓 Python 端可以讀到
    這個 component 高度為 0，使用者看不到。
    """
    components.html(f"""
    <script>
    (function() {{
        const KEY = "{LS_KEY}";
        let bid = localStorage.getItem(KEY);
        if (!bid) {{
            bid = crypto.randomUUID ? crypto.randomUUID()
                  : Math.random().toString(36).slice(2) + Date.now().toString(36);
            localStorage.setItem(KEY, bid);
        }}
        // 把 bid 寫進 URL query param，讓 Streamlit Python 端讀取
        const url = new URL(window.parent.location.href);
        if (url.searchParams.get("bid") !== bid) {{
            url.searchParams.set("bid", bid);
            window.parent.history.replaceState(null, "", url.toString());
            // 通知 Streamlit 重新讀取 query params
            window.parent.location.reload();
        }}
    }})();
    </script>
    """, height=0)


# ===========================================================================
# --- 2. 使用者股票清單：伺服器端 JSON（以 browser_id 區分）---
# ===========================================================================

def user_file(bid: str) -> str:
    """根據 browser_id 取得對應的 JSON 檔案路徑"""
    # 只保留安全字元，避免路徑注入
    safe_bid = "".join(c for c in bid if c.isalnum() or c in "-_")[:64]
    return os.path.join(USER_DATA_DIR, f"{safe_bid}.json")


def load_user_stocks(bid: str) -> list:
    path = user_file(bid)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return list(DEFAULT_STOCKS)


def save_user_stocks(bid: str, stocks: list):
    path = user_file(bid)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stocks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ===========================================================================
# --- 3. Telegram + FinMind Token（伺服器端共用）---
# ===========================================================================

def load_tg_config():
    if os.path.exists(TG_SAVE_FILE):
        try:
            with open(TG_SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"tg_token": "", "tg_chat_id": "", "tg_threshold": 3.0, "finmind_token": ""}


def save_tg_config():
    with open(TG_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "tg_token":      st.session_state.tg_token,
            "tg_chat_id":    st.session_state.tg_chat_id,
            "tg_threshold":  st.session_state.tg_threshold,
            "finmind_token": st.session_state.finmind_token,
        }, f, ensure_ascii=False, indent=4)


# ===========================================================================
# --- 4. session_state 初始化 ---
# ===========================================================================
if "initialized" not in st.session_state:
    tg_cfg = load_tg_config()
    st.session_state.update({
        "tg_token":      tg_cfg["tg_token"],
        "tg_chat_id":    tg_cfg["tg_chat_id"],
        "tg_threshold":  tg_cfg["tg_threshold"],
        "finmind_token": tg_cfg.get("finmind_token", ""),
        "initialized":   True,
        "alert_history": {},
        "hist_cache":    {},
        "my_stocks":     list(DEFAULT_STOCKS),
    })

# 取得 browser_id（從 URL query param 讀，JS 那邊會寫進去）
browser_id = st.query_params.get("bid", "")

# 如果 browser_id 有效，且本次 session 尚未從檔案載入，就載入
if browser_id and not st.session_state.get("stocks_loaded_bid") == browser_id:
    st.session_state.my_stocks = load_user_stocks(browser_id)
    st.session_state.stocks_loaded_bid = browser_id


# ===========================================================================
# --- 5. 歷史資料快取（yfinance）---
# ===========================================================================

def get_history_cached(stock_id: str) -> pd.DataFrame:
    cache = st.session_state.hist_cache
    today = today_str()
    if stock_id in cache and cache[stock_id]["cached_date"] == today:
        return cache[stock_id]["df"].copy()

    df = pd.DataFrame()
    for suffix in [".TW", ".TWO"]:
        try:
            temp = yf.download(f"{stock_id}{suffix}", period="6mo", progress=False)
            if not temp.empty:
                df = temp
                break
        except Exception:
            continue
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.astype(float).ffill()
    df.index = pd.to_datetime(df.index).normalize()
    yesterday = pd.Timestamp(today) - timedelta(days=1)
    df = df[df.index <= yesterday]
    cache[stock_id] = {"df": df, "cached_date": today}
    return df.copy()


# ===========================================================================
# --- 6. 即時資料（FinMind）+ 縫合 ---
# ===========================================================================

def get_finmind_today(stock_id: str):
    try:
        dl = DataLoader()
        token = st.session_state.get("finmind_token", "")
        if token:
            dl.login_by_token(api_token=token)
        today = today_str()
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=today, end_date=today)
        if df is None or df.empty:
            return None
        row = df.iloc[-1]
        return pd.Series({
            "Open":   float(row.get("open",   row.get("Open",   0))),
            "High":   float(row.get("max",    row.get("High",   0))),
            "Low":    float(row.get("min",    row.get("Low",    0))),
            "Close":  float(row.get("close",  row.get("Close",  0))),
            "Volume": float(row.get("volume", row.get("Volume", 0))),
        }, name=pd.Timestamp(today))
    except Exception as e:
        st.warning(f"FinMind 即時抓取失敗（{stock_id}）：{e}")
        return None


def stitch_dataframe(hist_df: pd.DataFrame, today_row) -> tuple:
    if today_row is not None and is_market_open():
        today_df = pd.DataFrame([today_row])
        today_df.index.name = hist_df.index.name
        merged = pd.concat([hist_df, today_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged, "📡 FinMind 即時縫合"
    return hist_df, "🗂 yfinance 歷史"


# ===========================================================================
# --- 7. KD 金叉判斷 ---
# ===========================================================================

def classify_kd_cross(k_now, d_now, k_prev, d_prev):
    if not ((k_prev <= d_prev) and (k_now > d_now)):
        return False, ""
    if (k_now - d_now) < 1.0:
        return False, ""
    avg = (k_now + d_now) / 2
    if avg < 20:
        return True, "✅ KD 低檔金叉（超賣區，可靠度高）"
    elif avg < 80:
        return True, "✅ KD 標準金叉（中段，偏多）"
    return False, ""


# ===========================================================================
# --- 8. 技術指標計算 ---
# ===========================================================================

def calc_indicators(df: pd.DataFrame):
    if len(df) < 30:
        return None
    close = pd.Series(df["Close"].values.flatten(), index=df.index).astype(float)
    high  = pd.Series(df["High"].values.flatten(),  index=df.index).astype(float)
    low   = pd.Series(df["Low"].values.flatten(),   index=df.index).astype(float)
    try:
        try:
            df = df.copy()
            df["MA5"]       = SMAIndicator(close, window=5).sma_indicator()
            df["MA10"]      = SMAIndicator(close, window=10).sma_indicator()
            df["MA20"]      = SMAIndicator(close, window=20).sma_indicator()
            stoch           = StochasticOscillator(high, low, close, window=9)
            df["K"]         = stoch.stoch()
            df["D"]         = stoch.stoch_signal()
            df["MACD_diff"] = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
            df["RSI"]       = RSIIndicator(close, window=14).rsi()
            df["BBM"]       = BollingerBands(close, window=20).bollinger_mavg()
        except Exception:
            df = df.copy()
            df["MA5"]       = SMAIndicator(close, n=5).sma_indicator()
            df["MA10"]      = SMAIndicator(close, n=10).sma_indicator()
            df["MA20"]      = SMAIndicator(close, n=20).sma_indicator()
            stoch           = StochasticOscillator(high, low, close, n=9)
            df["K"]         = stoch.stoch()
            df["D"]         = stoch.stoch_signal()
            df["MACD_diff"] = MACD(close, n_slow=26, n_fast=12, n_sign=9).macd_diff()
            df["RSI"]       = RSIIndicator(close, n=14).rsi()
            df["BBM"]       = BollingerBands(close, n=20).bollinger_mavg()
        return df
    except Exception:
        return None


# ===========================================================================
# --- 9. 主分析函數 ---
# ===========================================================================

@st.cache_data(ttl=60)
def fetch_and_analyze(stock_id: str):
    hist_df = get_history_cached(stock_id)
    if hist_df.empty:
        return None
    today_row = get_finmind_today(stock_id) if is_market_open() else None
    df, source = stitch_dataframe(hist_df, today_row)
    df = calc_indicators(df)
    if df is None:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score, details = 0, []

    if last["MA5"] > last["MA10"] > last["MA20"]:
        details.append("✅ 均線多頭排列"); score += 1
    kd_ok, kd_lbl = classify_kd_cross(float(last["K"]), float(last["D"]),
                                       float(prev["K"]), float(prev["D"]))
    if kd_ok:
        details.append(kd_lbl); score += 1
    if last["MACD_diff"] > 0:
        details.append("✅ MACD 柱狀體轉正"); score += 1
    if last["RSI"] > 50:
        details.append("✅ RSI 強勢區"); score += 1
    if last["Close"] > last["BBM"]:
        details.append("✅ 站穩月線(MA20)"); score += 1

    dm = {
        5: ("S (極強)", "🔥 續抱/加碼",   "red"),
        4: ("A (強勢)", "🚀 偏多持股",   "orange"),
        3: ("B (轉強)", "📈 少量試單",   "green"),
        2: ("C (盤整)", "⚖️ 暫時觀望",  "blue"),
        1: ("D (弱勢)", "📉 減碼避險",   "gray"),
        0: ("E (極弱)", "🚫 觀望不進場", "black"),
    }
    grade, action, color = dm[score]
    return {
        "price": float(last["Close"]),
        "pct":   (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100,
        "grade": grade, "action": action, "color": color,
        "details": details, "score": score,
        "k": float(last["K"]), "d": float(last["D"]),
        "source": source,
    }


# ===========================================================================
# --- 10. 介面 ---
# ===========================================================================
st.set_page_config(page_title="台股決策系統 V7.3", layout="centered")
st.title("🤖 賓哥的 AI 技術決策支援")

# 注入 browser_id JS（高度 0，使用者看不到）
# 只在尚未取得 browser_id 時執行，避免無限 reload
if not browser_id:
    get_browser_id_component()
    st.info("⏳ 初始化中，請稍候...")
    st.stop()

# 開盤狀態
if is_market_open():
    st.success("🟢 **開盤中** — 歷史快取來自 yfinance，今日即時棒由 FinMind 縫合")
else:
    st.info(f"🔵 **非開盤時間**（{now_tw().strftime('%H:%M')}）— 使用 yfinance 歷史快取")

st.caption(f"📌 您的專屬清單已儲存於此瀏覽器，重新整理或關閉後仍會保留。")

# ── 新增自選股票 ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("🔍 新增自選股票")
    c1, c2, c3 = st.columns([2, 3, 1.2])
    input_id   = c1.text_input("代號", key="add_id")
    input_name = c2.text_input("名稱", key="add_name")
    if c3.button("➕ 新增", use_container_width=True):
        if input_id and input_name:
            if not any(s["id"] == input_id for s in st.session_state.my_stocks):
                st.session_state.my_stocks.append({"id": input_id, "name": input_name})
                save_user_stocks(browser_id, st.session_state.my_stocks)
                st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("📡 FinMind")
    st.session_state.finmind_token = st.text_input(
        "API Token（選填）", type="password",
        value=st.session_state.finmind_token,
        help="未填使用免費版，開盤時有速率限制。",
    )

    st.subheader("🔔 Telegram 通知")
    st.session_state.tg_token     = st.text_input("Bot Token", type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id   = st.text_input("Chat ID", value=st.session_state.tg_chat_id)
    st.session_state.tg_threshold = st.number_input("通知門檻 (%)", value=st.session_state.tg_threshold)

    if st.button("💾 儲存並刷新"):
        save_tg_config()
        st.cache_data.clear()
        st.rerun()

    st.divider()

    if st.button("🚀 手動掃描並發送通知", use_container_width=True):
        st.cache_data.clear()
        found = 0
        for s in st.session_state.my_stocks:
            res = fetch_and_analyze(s["id"])
            if res and abs(res["pct"]) >= st.session_state.tg_threshold:
                msg = (
                    f"🔔 <b>【AI 決策通知】</b>\n\n"
                    f"標的：<b>{s['name']} ({s['id']})</b>\n"
                    f"目前股價：<b>{res['price']:.2f}</b>\n"
                    f"今日漲跌：<b>{res['pct']:+.2f}%</b>\n"
                    f"技術評級：{res['grade']}\n"
                    f"建議決策：<b>{res['action']}</b>\n\n"
                    f"符合指標：{', '.join(res['details']) if res['details'] else '無'}"
                )
                requests.post(
                    f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage",
                    json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"},
                )
                found += 1
        st.success(f"掃描完成，已發送 {found} 則通知")

    st.divider()
    with st.expander("📖 資料架構說明"):
        st.markdown("""
**歷史快取（yfinance）**：啟動或跨日時抓近 6 個月，同日內不重抓。

**即時縫合（FinMind）**：開盤中只抓今天這一棒接在歷史尾端，ttl=60s。

**非開盤**：完全不呼叫 FinMind。
        """)
    with st.expander("📖 KD 金叉說明"):
        st.markdown("""
1. 真實交叉：前 K ≤ D，本根 K > D
2. 幅度 ≥ 1（排除噪音）
3. KD < 20 → 低檔金叉 ✅ / 20~79 → 標準 ✅ / ≥ 80 → 高檔鈍化 ❌
        """)

# ── 股票清單 ──────────────────────────────────────────────────────────────
st.divider()
for idx, stock in enumerate(st.session_state.my_stocks):
    res = fetch_and_analyze(stock["id"])
    if res:
        with st.container(border=True):
            col_info, col_metric, col_del = st.columns([3, 2, 0.6])
            with col_info:
                st.write(f"### {stock['name']} ({stock['id']})")
                st.caption(f"資料來源：{res['source']}")
                st.markdown(f"評級：`{res['grade']}`")
                st.markdown(
                    f"**建議決策：<span style='color:{res['color']}'>{res['action']}</span>**",
                    unsafe_allow_html=True,
                )
                st.markdown(f"符合指標：{'　'.join(res['details']) if res['details'] else '無'}")
                st.caption(f"KD 值：K={res['k']:.1f} / D={res['d']:.1f}")
            with col_metric:
                st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col_del:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_user_stocks(browser_id, st.session_state.my_stocks)
                    st.rerun()
    else:
        with st.container(border=True):
            st.warning(f"⚠️ **{stock['name']} ({stock['id']})** 資料抓取失敗。")

if st.button("🔄 全部重新整理"):
    st.cache_data.clear()
    st.rerun()
