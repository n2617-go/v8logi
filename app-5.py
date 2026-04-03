import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
import os
from datetime import datetime, time as dt_time, timedelta
from FinMind.data import DataLoader
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# ── Cookie 管理（每個瀏覽器獨立清單）───────────────────────────────────────
try:
    from streamlit_cookies_manager import CookieManager
    cookies = CookieManager()
    if not cookies.ready():
        st.stop()
    COOKIE_AVAILABLE = True
except ImportError:
    COOKIE_AVAILABLE = False

# ===========================================================================
# --- 0. 基礎設定 ---
# ===========================================================================
tw_tz        = pytz.timezone('Asia/Taipei')
MARKET_OPEN  = dt_time(9, 0)
MARKET_CLOSE = dt_time(13, 30)
COOKIE_KEY   = "my_stocks_v1"
TG_SAVE_FILE = "tg_config.json"
DEFAULT_STOCKS = [{"id": "2330", "name": "台積電"}]


def is_market_open() -> bool:
    """判斷目前是否為台股開盤時間（平日 09:00–13:30）"""
    now_tw = datetime.now(tw_tz)
    if now_tw.weekday() >= 5:   # 週六、週日
        return False
    t = now_tw.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


# ── Cookie 讀寫 ─────────────────────────────────────────────────────────────
def load_stocks_from_cookie():
    if not COOKIE_AVAILABLE:
        return list(DEFAULT_STOCKS)
    raw = cookies.get(COOKIE_KEY)
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return list(DEFAULT_STOCKS)


def save_stocks_to_cookie(stocks):
    if COOKIE_AVAILABLE:
        cookies[COOKIE_KEY] = json.dumps(stocks, ensure_ascii=False)
        cookies.save()


# ── Telegram + FinMind Token 設定（伺服器端共用）──────────────────────────
def load_tg_config():
    if os.path.exists(TG_SAVE_FILE):
        try:
            with open(TG_SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"tg_token": "", "tg_chat_id": "", "tg_threshold": 3.0, "finmind_token": ""}


def save_tg_config():
    data = {
        "tg_token":      st.session_state.tg_token,
        "tg_chat_id":    st.session_state.tg_chat_id,
        "tg_threshold":  st.session_state.tg_threshold,
        "finmind_token": st.session_state.finmind_token,
    }
    with open(TG_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ── session_state 初始化 ─────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    tg_cfg = load_tg_config()
    st.session_state.update({
        "my_stocks":     load_stocks_from_cookie(),
        "tg_token":      tg_cfg["tg_token"],
        "tg_chat_id":    tg_cfg["tg_chat_id"],
        "tg_threshold":  tg_cfg["tg_threshold"],
        "finmind_token": tg_cfg.get("finmind_token", ""),
        "initialized":   True,
        "alert_history": {},
    })


# ===========================================================================
# --- 1. 資料抓取：FinMind（開盤中）/ yfinance（非開盤）---
# ===========================================================================

def fetch_finmind_intraday(stock_id: str) -> pd.DataFrame:
    """
    用 FinMind 抓近 6 個月台股日K資料（含今日即時最新一筆）。
    回傳 Open / High / Low / Close / Volume，index 為日期。
    失敗時回傳空 DataFrame。
    """
    try:
        dl = DataLoader()
        token = st.session_state.get("finmind_token", "")
        if token:
            dl.login_by_token(api_token=token)

        today     = datetime.now(tw_tz).strftime("%Y-%m-%d")
        start_day = (datetime.now(tw_tz) - timedelta(days=180)).strftime("%Y-%m-%d")

        df = dl.taiwan_stock_daily(
            stock_id   = stock_id,
            start_date = start_day,
            end_date   = today,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # FinMind 欄位名稱映射
        df = df.rename(columns={
            "date":   "Date",
            "open":   "Open",
            "max":    "High",
            "min":    "Low",
            "close":  "Close",
            "volume": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df

    except Exception as e:
        st.warning(f"FinMind 抓取失敗（{stock_id}）：{e}，自動切換至 yfinance。")
        return pd.DataFrame()


def fetch_yfinance_history(stock_id: str) -> pd.DataFrame:
    """
    用 yfinance 抓近 6 個月歷史日K。
    回傳含 Open / High / Low / Close 的 DataFrame，index 為日期。
    """
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
    return df.astype(float).ffill()


def fetch_ohlc(stock_id: str) -> tuple:
    """
    依開盤狀態選擇資料來源，回傳 (df, source_label)。
    開盤中優先 FinMind，失敗才 fallback yfinance；
    非開盤直接用 yfinance。
    """
    if is_market_open():
        df = fetch_finmind_intraday(stock_id)
        if not df.empty:
            return df, "📡 FinMind 即時"
        df = fetch_yfinance_history(stock_id)
        return df, "📡 yfinance（FinMind 備援）"
    else:
        df = fetch_yfinance_history(stock_id)
        return df, "🗂 yfinance 歷史"


# ===========================================================================
# --- 2. KD 黃金交叉判斷（改良版）---
# ===========================================================================

def classify_kd_cross(k_now, d_now, k_prev, d_prev):
    """
    回傳 (is_valid: bool, label: str)

    ① 真實交叉：前一根 K ≤ D，本根 K > D（排除長期貼合假叉）
    ② 交叉幅度 ≥ 1（排除噪音微叉）
    ③ 依 KD 區域：
       KD < 20  → 低檔金叉，最可靠  ✅
       20~79    → 標準金叉          ✅
       KD ≥ 80  → 高檔鈍化，不計分  ❌
    """
    real_cross = (k_prev <= d_prev) and (k_now > d_now)
    if not real_cross:
        return False, ""

    if (k_now - d_now) < 1.0:
        return False, ""

    avg_kd = (k_now + d_now) / 2
    if avg_kd < 20:
        return True, "✅ KD 低檔金叉（超賣區，可靠度高）"
    elif avg_kd < 80:
        return True, "✅ KD 標準金叉（中段，偏多）"
    else:
        return False, ""   # 高檔鈍化，不計分


# ===========================================================================
# --- 3. 分析與決策引擎 ---
# ===========================================================================

@st.cache_data(ttl=60)
def fetch_and_analyze(stock_id: str):
    df, source = fetch_ohlc(stock_id)
    if df is None or df.empty:
        return None

    close = pd.Series(df["Close"].values.flatten(), index=df.index).astype(float)
    high  = pd.Series(df["High"].values.flatten(),  index=df.index).astype(float)
    low   = pd.Series(df["Low"].values.flatten(),   index=df.index).astype(float)

    if len(close) < 30:
        return None

    try:
        try:
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
            df["MA5"]       = SMAIndicator(close, n=5).sma_indicator()
            df["MA10"]      = SMAIndicator(close, n=10).sma_indicator()
            df["MA20"]      = SMAIndicator(close, n=20).sma_indicator()
            stoch           = StochasticOscillator(high, low, close, n=9)
            df["K"]         = stoch.stoch()
            df["D"]         = stoch.stoch_signal()
            df["MACD_diff"] = MACD(close, n_slow=26, n_fast=12, n_sign=9).macd_diff()
            df["RSI"]       = RSIIndicator(close, n=14).rsi()
            df["BBM"]       = BollingerBands(close, n=20).bollinger_mavg()
    except Exception:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score   = 0
    details = []

    # 均線多頭排列
    if last["MA5"] > last["MA10"] > last["MA20"]:
        details.append("✅ 均線多頭排列")
        score += 1

    # KD 黃金交叉（改良版）
    kd_valid, kd_label = classify_kd_cross(
        float(last["K"]), float(last["D"]),
        float(prev["K"]), float(prev["D"]),
    )
    if kd_valid:
        details.append(kd_label)
        score += 1

    # MACD
    if last["MACD_diff"] > 0:
        details.append("✅ MACD 柱狀體轉正")
        score += 1

    # RSI
    if last["RSI"] > 50:
        details.append("✅ RSI 強勢區")
        score += 1

    # 站穩月線
    if last["Close"] > last["BBM"]:
        details.append("✅ 站穩月線(MA20)")
        score += 1

    decision_map = {
        5: {"grade": "S (極強)", "action": "🔥 續抱/加碼",   "color": "red"},
        4: {"grade": "A (強勢)", "action": "🚀 偏多持股",   "color": "orange"},
        3: {"grade": "B (轉強)", "action": "📈 少量試單",   "color": "green"},
        2: {"grade": "C (盤整)", "action": "⚖️ 暫時觀望",  "color": "blue"},
        1: {"grade": "D (弱勢)", "action": "📉 減碼避險",   "color": "gray"},
        0: {"grade": "E (極弱)", "action": "🚫 觀望不進場", "color": "black"},
    }
    res = decision_map[score]

    return {
        "price":   float(last["Close"]),
        "pct":     (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100,
        "grade":   res["grade"],
        "action":  res["action"],
        "color":   res["color"],
        "details": details,
        "score":   score,
        "k":       float(last["K"]),
        "d":       float(last["D"]),
        "source":  source,
    }


# ===========================================================================
# --- 4. 介面 ---
# ===========================================================================
st.set_page_config(page_title="台股決策系統 V7.2", layout="centered")
st.title("🤖 台股 AI 技術分級決策支援")

# 開盤狀態 Banner
if is_market_open():
    st.success("🟢 **開盤中** — 使用 FinMind 即時資料計算技術指標")
else:
    now_str = datetime.now(tw_tz).strftime("%H:%M")
    st.info(f"🔵 **非開盤時間**（{now_str}）— 使用 yfinance 歷史資料，不呼叫 FinMind")

if COOKIE_AVAILABLE:
    st.caption("📌 您的股票清單儲存於此瀏覽器 Cookie，不同瀏覽器／裝置為獨立清單。")
else:
    st.warning("⚠️ 請安裝 `extra-streamlit-components` 以啟用個人清單：`pip install extra-streamlit-components`")

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
                save_stocks_to_cookie(st.session_state.my_stocks)
                st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")

    st.subheader("📡 FinMind")
    st.session_state.finmind_token = st.text_input(
        "API Token（選填）",
        type="password",
        value=st.session_state.finmind_token,
        help="未填使用免費版，開盤時有速率限制；填入 Token 可提高上限。",
    )

    st.subheader("🔔 Telegram 通知")
    st.session_state.tg_token     = st.text_input("Bot Token",     type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id   = st.text_input("Chat ID",       value=st.session_state.tg_chat_id)
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
                url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
                requests.post(url, json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"})
                found += 1
        st.success(f"掃描完成，已發送 {found} 則通知")

    st.divider()
    with st.expander("📖 資料來源說明"):
        st.markdown("""
**開盤中（平日 09:00–13:30）**
- 優先使用 **FinMind** 抓取含今日即時資料的日K
- 若 FinMind 失敗，自動 fallback 至 yfinance

**非開盤時間**
- 直接使用 **yfinance** 抓歷史資料，不呼叫 FinMind
        """)

    with st.expander("📖 KD 金叉判斷說明"):
        st.markdown("""
**同時滿足以下條件才計分：**

1. **真實交叉**：前一根 K ≤ D，本根 K > D
2. **幅度 ≥ 1**：排除 0.x 差距的噪音假叉
3. **依 KD 位置分級**：
   - KD < 20 → ✅ 低檔金叉（超賣區，最可靠）
   - 20 ≤ KD < 80 → ✅ 標準金叉（偏多）
   - KD ≥ 80 → ❌ 高檔鈍化，**不計分**
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
                indicators = "　".join(res["details"]) if res["details"] else "無"
                st.markdown(f"符合指標：{indicators}")
                st.caption(f"KD 值：K={res['k']:.1f} / D={res['d']:.1f}")
            with col_metric:
                st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col_del:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_stocks_to_cookie(st.session_state.my_stocks)
                    st.rerun()
    else:
        with st.container(border=True):
            st.warning(f"⚠️ **{stock['name']} ({stock['id']})** 資料抓取失敗，請確認代號或稍後再試。")

if st.button("🔄 全部重新整理"):
    st.cache_data.clear()
    st.rerun()
