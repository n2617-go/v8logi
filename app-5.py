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
    import extra_streamlit_components as stx
    cookie_manager = stx.CookieManager()
    COOKIE_AVAILABLE = True
except ImportError:
    COOKIE_AVAILABLE = False

# ===========================================================================
# --- 0. 基礎設定 ---
# ===========================================================================
tw_tz        = pytz.timezone("Asia/Taipei")
MARKET_OPEN  = dt_time(9, 0)
MARKET_CLOSE = dt_time(13, 30)
COOKIE_KEY   = "my_stocks_v1"
TG_SAVE_FILE = "tg_config.json"
DEFAULT_STOCKS = [{"id": "2330", "name": "台積電"}]


def now_tw() -> datetime:
    return datetime.now(tw_tz)


def is_market_open() -> bool:
    """平日 09:00–13:30 為開盤時間"""
    n = now_tw()
    if n.weekday() >= 5:
        return False
    return MARKET_OPEN <= n.time() <= MARKET_CLOSE


def today_str() -> str:
    return now_tw().strftime("%Y-%m-%d")


# ── Cookie 讀寫 ─────────────────────────────────────────────────────────────
def load_stocks_from_cookie():
    if not COOKIE_AVAILABLE:
        return list(DEFAULT_STOCKS)
    try:
        raw = cookie_manager.get(cookie=COOKIE_KEY)
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return list(DEFAULT_STOCKS)


def save_stocks_to_cookie(stocks):
    if COOKIE_AVAILABLE:
        try:
            cookie_manager.set(
                cookie=COOKIE_KEY,
                val=json.dumps(stocks, ensure_ascii=False),
                expires_at=datetime.now(tw_tz).replace(year=datetime.now(tw_tz).year + 1),
            )
        except Exception:
            pass


# ── Telegram + FinMind Token（伺服器端共用）──────────────────────────────
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


# ── session_state 初始化（非 cookie 部分，只跑一次）──────────────────────────
if "initialized" not in st.session_state:
    tg_cfg = load_tg_config()
    st.session_state.update({
        "my_stocks":      list(DEFAULT_STOCKS),  # 暫時預設，下方 cookie 同步會覆蓋
        "tg_token":       tg_cfg["tg_token"],
        "tg_chat_id":     tg_cfg["tg_chat_id"],
        "tg_threshold":   tg_cfg["tg_threshold"],
        "finmind_token":  tg_cfg.get("finmind_token", ""),
        "initialized":    True,
        "cookie_loaded":  False,   # 控制 cookie 只在 session 首次建立時讀取
        "alert_history":  {},
        "hist_cache":     {},
    })

# ── 首次載入時從 cookie 還原股票清單 ────────────────────────────────────────
# 用 cookie_loaded 旗標確保：
#   - 第一次載入（session 全新）→ 從 cookie 讀取並還原
#   - st.rerun() 觸發的重跑    → 跳過，保留 session_state 內已修改的清單
#   （避免 cookie 非同步寫入導致 rerun 時讀到舊值把清單蓋掉）
if not st.session_state.get("cookie_loaded", False):
    _cookie_stocks = load_stocks_from_cookie()
    if _cookie_stocks:
        st.session_state.my_stocks = _cookie_stocks
    st.session_state.cookie_loaded = True


# ===========================================================================
# --- 1. 歷史資料快取（yfinance）---
# ===========================================================================

def get_history_cached(stock_id: str) -> pd.DataFrame:
    """
    取得歷史日K快取。
    規則：
      - 快取不存在，或快取日期 != 今天（跨日了）→ 重新從 yfinance 抓取並存入快取
      - 快取存在且同日 → 直接回傳，不重抓
    回傳欄位：Open / High / Low / Close / Volume，index 為日期(date)
    """
    cache = st.session_state.hist_cache
    today = today_str()

    if stock_id in cache and cache[stock_id]["cached_date"] == today:
        return cache[stock_id]["df"].copy()

    # --- 重新抓取 yfinance 歷史（只到昨天，避免與今日即時重疊）---
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

    # 只保留「昨天以前」的資料，避免與 FinMind 今日即時棒重複
    yesterday = pd.Timestamp(today) - timedelta(days=1)
    df = df[df.index <= yesterday]

    cache[stock_id] = {"df": df, "cached_date": today}
    return df.copy()


# ===========================================================================
# --- 2. 即時資料（FinMind）→ 縫合 ---
# ===========================================================================

def get_finmind_today(stock_id: str) -> pd.Series | None:
    """
    用 FinMind 抓今天的即時一棒（Open/High/Low/Close/Volume）。
    回傳 pd.Series，index 為欄位名稱；失敗回傳 None。
    """
    try:
        dl = DataLoader()
        token = st.session_state.get("finmind_token", "")
        if token:
            dl.login_by_token(api_token=token)

        today = today_str()
        df = dl.taiwan_stock_daily(
            stock_id   = stock_id,
            start_date = today,
            end_date   = today,
        )
        if df is None or df.empty:
            return None

        row = df.iloc[-1]
        return pd.Series({
            "Open":   float(row.get("open",  row.get("Open",  0))),
            "High":   float(row.get("max",   row.get("High",  0))),
            "Low":    float(row.get("min",   row.get("Low",   0))),
            "Close":  float(row.get("close", row.get("Close", 0))),
            "Volume": float(row.get("volume",row.get("Volume",0))),
        }, name=pd.Timestamp(today))

    except Exception as e:
        st.warning(f"FinMind 即時抓取失敗（{stock_id}）：{e}")
        return None


def stitch_dataframe(hist_df: pd.DataFrame, today_row: pd.Series | None) -> tuple:
    """
    將歷史 DataFrame 與今日即時一棒縫合。
    - 開盤中且 today_row 存在：縫合今日棒，回傳 (merged_df, "FinMind 即時縫合")
    - 否則直接回傳歷史資料，回傳 (hist_df, "yfinance 歷史")
    """
    if today_row is not None and is_market_open():
        today_df = pd.DataFrame([today_row])
        today_df.index.name = hist_df.index.name
        merged = pd.concat([hist_df, today_df])
        merged = merged[~merged.index.duplicated(keep="last")]  # 避免重複 index
        merged = merged.sort_index()
        return merged, "📡 FinMind 即時縫合"
    else:
        return hist_df, "🗂 yfinance 歷史"


# ===========================================================================
# --- 3. KD 黃金交叉判斷（改良版）---
# ===========================================================================

def classify_kd_cross(k_now, d_now, k_prev, d_prev):
    """
    ① 真實交叉：前一根 K ≤ D，本根 K > D
    ② 幅度 ≥ 1（排除噪音微叉）
    ③ 依區域：KD<20 低檔金叉 ✅ / 20~79 標準金叉 ✅ / KD≥80 高檔鈍化 ❌
    """
    if not ((k_prev <= d_prev) and (k_now > d_now)):
        return False, ""
    if (k_now - d_now) < 1.0:
        return False, ""
    avg = (k_now + d_now) / 2
    if avg < 20:
        return True, "✅ KD 低檔金叉（超賣區，可靠度高）"
    elif avg < 80:
        return True, "✅ KD 標準金叉（中段，偏多）"
    else:
        return False, ""


# ===========================================================================
# --- 4. 技術指標計算 ---
# ===========================================================================

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """對傳入的 DataFrame 計算所有技術指標，回傳加工後的 df 或 None"""
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
# --- 5. 主分析函數（縫合 + 指標 + 評分）---
# ===========================================================================

# ttl=60：每分鐘最多重新呼叫一次 FinMind 抓今日棒
@st.cache_data(ttl=60)
def fetch_and_analyze(stock_id: str):
    # Step 1：取歷史快取（yfinance，跨日才重抓）
    hist_df = get_history_cached(stock_id)
    if hist_df.empty:
        return None

    # Step 2：開盤中才呼叫 FinMind 抓今日即時棒
    today_row = get_finmind_today(stock_id) if is_market_open() else None

    # Step 3：縫合
    df, source = stitch_dataframe(hist_df, today_row)

    # Step 4：計算指標
    df = calc_indicators(df)
    if df is None:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Step 5：評分
    score   = 0
    details = []

    if last["MA5"] > last["MA10"] > last["MA20"]:
        details.append("✅ 均線多頭排列"); score += 1

    kd_ok, kd_lbl = classify_kd_cross(
        float(last["K"]), float(last["D"]),
        float(prev["K"]), float(prev["D"]),
    )
    if kd_ok:
        details.append(kd_lbl); score += 1

    if last["MACD_diff"] > 0:
        details.append("✅ MACD 柱狀體轉正"); score += 1
    if last["RSI"] > 50:
        details.append("✅ RSI 強勢區"); score += 1
    if last["Close"] > last["BBM"]:
        details.append("✅ 站穩月線(MA20)"); score += 1

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
# --- 6. 介面 ---
# ===========================================================================
st.set_page_config(page_title="台股決策系統 V7.3", layout="centered")
st.title("🤖 台股 AI 技術分級決策支援")

# 開盤狀態 Banner
if is_market_open():
    st.success("🟢 **開盤中** — 歷史資料來自 yfinance 快取，今日即時棒由 FinMind 縫合更新")
else:
    st.info(f"🔵 **非開盤時間**（{now_tw().strftime('%H:%M')}）— 使用 yfinance 歷史快取，不呼叫 FinMind")

if COOKIE_AVAILABLE:
    st.caption("📌 股票清單儲存於此瀏覽器 Cookie，不同瀏覽器／裝置為獨立清單。")
else:
    st.warning("⚠️ 請安裝 `extra-streamlit-components`：`pip install extra-streamlit-components`")

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
        help="未填使用免費版，開盤時有速率限制；填入 Token 可大幅提升上限。",
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
                requests.post(
                    f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage",
                    json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"},
                )
                found += 1
        st.success(f"掃描完成，已發送 {found} 則通知")

    st.divider()

    with st.expander("📖 資料架構說明"):
        st.markdown("""
**歷史快取（yfinance）**
- 啟動或跨日時抓取近 6 個月日K，存入 `session_state`
- 同一天內不重抓，直接讀取記憶體快取

**即時縫合（FinMind）**
- 開盤中（09:00–13:30）才呼叫 FinMind
- 只抓「今天」這一棒，接在歷史資料尾端
- `@st.cache_data(ttl=60)` 控制每 60 秒最多縫合一次

**非開盤時間**
- 完全不呼叫 FinMind，節省 API 配額
        """)

    with st.expander("📖 KD 金叉判斷說明"):
        st.markdown("""
**同時滿足以下條件才計分：**
1. 真實交叉：前一根 K ≤ D，本根 K > D
2. 幅度 ≥ 1（排除噪音假叉）
3. 依 KD 位置：
   - KD < 20 → ✅ 低檔金叉（超賣，最可靠）
   - 20 ≤ KD < 80 → ✅ 標準金叉
   - KD ≥ 80 → ❌ 高檔鈍化，不計分
        """)

# ── 股票清單顯示 ──────────────────────────────────────────────────────────
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
