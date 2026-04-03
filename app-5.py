import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
import os
from datetime import datetime, time as dt_time
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# --- Cookie 管理 ---
try:
    from streamlit_cookies_manager import EncryptedCookieManager, CookieManager
    cookies = CookieManager()   # 不加密，直接明文 cookie
    if not cookies.ready():
        st.stop()
    COOKIE_AVAILABLE = True
except ImportError:
    COOKIE_AVAILABLE = False

# --- 0. 基礎設定 ---
tw_tz = pytz.timezone('Asia/Taipei')
DEFAULT_STOCKS = [{"id": "2330", "name": "台積電"}]
DEFAULT_CONFIG  = {"stocks": DEFAULT_STOCKS, "tg_token": "", "tg_chat_id": "", "tg_threshold": 3.0}
COOKIE_KEY = "my_stocks_v1"   # cookie 中存股票清單用的 key


# ── 從 cookie 讀取使用者自己的股票清單 ──────────────────────────────────────
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


# ── Telegram 設定仍存伺服器端（各使用者共用，或可改為各自 cookie）──────────
TG_SAVE_FILE = "tg_config.json"

def load_tg_config():
    if os.path.exists(TG_SAVE_FILE):
        try:
            with open(TG_SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"tg_token": "", "tg_chat_id": "", "tg_threshold": 3.0}

def save_tg_config():
    data = {
        "tg_token":    st.session_state.tg_token,
        "tg_chat_id":  st.session_state.tg_chat_id,
        "tg_threshold":st.session_state.tg_threshold,
    }
    with open(TG_SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ── session_state 初始化 ─────────────────────────────────────────────────────
if 'initialized' not in st.session_state:
    tg_cfg = load_tg_config()
    st.session_state.update({
        'my_stocks':   load_stocks_from_cookie(),
        'tg_token':    tg_cfg["tg_token"],
        'tg_chat_id':  tg_cfg["tg_chat_id"],
        'tg_threshold':tg_cfg["tg_threshold"],
        'initialized': True,
        'alert_history': {},
    })


# ===========================================================================
# --- 1. KD 黃金交叉判斷邏輯（改良版）---
# ===========================================================================
def classify_kd_cross(k_now, d_now, k_prev, d_prev):
    """
    回傳 (is_valid_cross: bool, label: str)
    判斷規則：
    ① 必要條件：K 從「低於 D」翻轉到「高於 D」（真實交叉，不接受長期 K > D 的貼合）
    ② 交叉幅度必須 >= 1（避免 0.1 差距的假叉）
    ③ 依 KD 所在區域分三種情況：
       - KD < 20（超賣區）：✅ 低檔金叉，最可靠
       - 20 ≤ KD < 80（中段）：✅ 標準金叉，可接受
       - KD ≥ 80（超買區）：❌ 高檔鈍化金叉，跳過不計分
    ④ 排除「長期鈍化」：若前 2 日 K 持續 > D 且當下非剛剛翻越，視為鈍化不計
    """
    # 必要條件①：本次是真實交叉（前一根 K 在 D 以下，現在穿越到 D 以上）
    real_cross = (k_prev <= d_prev) and (k_now > d_now)

    if not real_cross:
        return False, ""

    # 必要條件②：交叉幅度（避免微小噪音）
    gap = k_now - d_now
    if gap < 1.0:
        return False, ""

    # 依區域判斷
    avg_kd = (k_now + d_now) / 2

    if avg_kd < 20:
        return True, "✅ KD 低檔金叉（超賣區，可靠度高）"
    elif avg_kd < 80:
        return True, "✅ KD 標準金叉（中段，偏多）"
    else:
        # 高檔金叉鈍化，不計分
        return False, ""


# ===========================================================================
# --- 2. 分析與決策引擎 ---
# ===========================================================================
@st.cache_data(ttl=60)
def fetch_and_analyze(stock_id):
    df = pd.DataFrame()
    for suffix in [".TW", ".TWO"]:
        try:
            temp_df = yf.download(f"{stock_id}{suffix}", period="6mo", progress=False)
            if not temp_df.empty:
                df = temp_df
                break
        except Exception:
            continue
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.astype(float).ffill()

    close = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)
    high  = pd.Series(df['High'].values.flatten(),  index=df.index).astype(float)
    low   = pd.Series(df['Low'].values.flatten(),   index=df.index).astype(float)

    try:
        try:
            df['MA5']  = SMAIndicator(close, window=5).sma_indicator()
            df['MA10'] = SMAIndicator(close, window=10).sma_indicator()
            df['MA20'] = SMAIndicator(close, window=20).sma_indicator()
            stoch = StochasticOscillator(high, low, close, window=9)
            df['K'] = stoch.stoch()
            df['D'] = stoch.stoch_signal()
            df['MACD_diff'] = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
            df['RSI']  = RSIIndicator(close, window=14).rsi()
            df['BBM']  = BollingerBands(close, window=20).bollinger_mavg()
        except Exception:
            df['MA5']  = SMAIndicator(close, n=5).sma_indicator()
            df['MA10'] = SMAIndicator(close, n=10).sma_indicator()
            df['MA20'] = SMAIndicator(close, n=20).sma_indicator()
            stoch = StochasticOscillator(high, low, close, n=9)
            df['K'] = stoch.stoch()
            df['D'] = stoch.stoch_signal()
            df['MACD_diff'] = MACD(close, n_slow=26, n_fast=12, n_sign=9).macd_diff()
            df['RSI']  = RSIIndicator(close, n=14).rsi()
            df['BBM']  = BollingerBands(close, n=20).bollinger_mavg()
    except Exception:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score   = 0
    details = []

    # 均線多頭排列
    if last['MA5'] > last['MA10'] > last['MA20']:
        details.append("✅ 均線多頭排列")
        score += 1

    # KD 黃金交叉（改良版邏輯）
    kd_valid, kd_label = classify_kd_cross(
        k_now  = float(last['K']),
        d_now  = float(last['D']),
        k_prev = float(prev['K']),
        d_prev = float(prev['D']),
    )
    if kd_valid:
        details.append(kd_label)
        score += 1

    # MACD
    if last['MACD_diff'] > 0:
        details.append("✅ MACD 柱狀體轉正")
        score += 1

    # RSI
    if last['RSI'] > 50:
        details.append("✅ RSI 強勢區")
        score += 1

    # 站穩月線
    if last['Close'] > last['BBM']:
        details.append("✅ 站穩月線(MA20)")
        score += 1

    decision_map = {
        5: {"grade": "S (極強)", "action": "🔥 續抱/加碼",    "color": "red"},
        4: {"grade": "A (強勢)", "action": "🚀 偏多持股",    "color": "orange"},
        3: {"grade": "B (轉強)", "action": "📈 少量試單",    "color": "green"},
        2: {"grade": "C (盤整)", "action": "⚖️ 暫時觀望",   "color": "blue"},
        1: {"grade": "D (弱勢)", "action": "📉 減碼避險",    "color": "gray"},
        0: {"grade": "E (極弱)", "action": "🚫 觀望不進場",  "color": "black"},
    }
    res = decision_map[score]

    return {
        "price":   float(last['Close']),
        "pct":     (float(last['Close']) - float(prev['Close'])) / float(prev['Close']) * 100,
        "grade":   res["grade"],
        "action":  res["action"],
        "color":   res["color"],
        "details": details,
        "score":   score,
        # 額外回傳 KD 值供顯示
        "k": float(last['K']),
        "d": float(last['D']),
    }


# ===========================================================================
# --- 3. 介面 ---
# ===========================================================================
st.set_page_config(page_title="台股決策系統 V7.2", layout="centered")
st.title("🤖 台股 AI 技術分級決策支援")

if COOKIE_AVAILABLE:
    st.caption("📌 您的股票清單已儲存在此瀏覽器，換個瀏覽器或裝置會是獨立清單。")
else:
    st.warning("⚠️ 未安裝 `extra-streamlit-components`，股票清單無法跨分頁獨立儲存。請執行：`pip install extra-streamlit-components`")

# ── 新增自選股票 ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("🔍 新增自選股票")
    c1, c2, c3 = st.columns([2, 3, 1.2])
    input_id   = c1.text_input("代號", key="add_id")
    input_name = c2.text_input("名稱", key="add_name")
    if c3.button("➕ 新增", use_container_width=True):
        if input_id and input_name:
            if not any(s['id'] == input_id for s in st.session_state.my_stocks):
                st.session_state.my_stocks.append({"id": input_id, "name": input_name})
                save_stocks_to_cookie(st.session_state.my_stocks)
                st.rerun()

# ── Sidebar：Telegram 通知設定 ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 通知設定")
    st.session_state.tg_token     = st.text_input("Bot Token",     type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id   = st.text_input("Chat ID",       value=st.session_state.tg_chat_id)
    st.session_state.tg_threshold = st.number_input("通知門檻 (%)", value=st.session_state.tg_threshold)
    if st.button("💾 儲存並刷新"):
        save_tg_config()
        st.cache_data.clear()
        st.rerun()
    st.divider()

    if st.button("🚀 手動測試掃描並發送通知", use_container_width=True):
        st.cache_data.clear()
        found = 0
        for s in st.session_state.my_stocks:
            res = fetch_and_analyze(s['id'])
            if res and abs(res['pct']) >= st.session_state.tg_threshold:
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
    # KD 邏輯說明
    with st.expander("📖 KD 金叉判斷說明"):
        st.markdown("""
**新版判斷條件（同時滿足才計分）：**

1. **真實交叉**：前一根 K ≤ D，本根 K > D（排除長期貼合）
2. **交叉幅度** ≥ 1（排除噪音假叉）
3. **依 KD 位置分級**：
   - KD < 20 → ✅ 低檔金叉（超賣區，可靠度最高）
   - 20 ≤ KD < 80 → ✅ 標準金叉（偏多）
   - KD ≥ 80 → ❌ 高檔鈍化，**不計分**
        """)

# ── 股票清單顯示 ─────────────────────────────────────────────────────────────
st.divider()
for idx, stock in enumerate(st.session_state.my_stocks):
    res = fetch_and_analyze(stock['id'])
    if res:
        with st.container(border=True):
            col_info, col_metric, col_del = st.columns([3, 2, 0.6])
            with col_info:
                st.write(f"### {stock['name']} ({stock['id']})")
                st.markdown(f"評級：`{res['grade']}`")
                st.markdown(
                    f"**建議決策：<span style='color:{res['color']}'>{res['action']}</span>**",
                    unsafe_allow_html=True,
                )
                indicators = "　".join(res['details']) if res['details'] else "無"
                st.markdown(f"符合指標：{indicators}")
                # 顯示 KD 目前數值
                st.caption(f"KD 值：K={res['k']:.1f} / D={res['d']:.1f}")
            with col_metric:
                st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col_del:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_stocks_to_cookie(st.session_state.my_stocks)
                    st.rerun()

if st.button("🔄 全部重新整理"):
    st.cache_data.clear()
    st.rerun()
