
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
from datetime import datetime
from streamlit_cookies_manager import CookieManager
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# --- 0. 基礎設定 ---
st.set_page_config(page_title="台股決策系統 V7.3", layout="centered")

# 1. 先初始化 session_state (避免 Cookie 延遲導致清單消失)
if 'my_stocks' not in st.session_state:
    st.session_state.my_stocks = [{"id": "2330", "name": "台積電"}]
if 'tg_token' not in st.session_state:
    st.session_state.tg_token = ""
if 'tg_chat_id' not in st.session_state:
    st.session_state.tg_chat_id = ""
if 'tg_threshold' not in st.session_state:
    st.session_state.tg_threshold = 3.0

# 2. Cookie 管理
cookies = CookieManager(prefix="twstock_")
if not cookies.ready():
    st.info("正在連線至瀏覽器儲存區...")
    st.stop()

# 3. 當 Cookie 準備好，嘗試從 Cookie 覆蓋 session_state (僅執行一次)
if 'cookie_loaded' not in st.session_state:
    raw_stocks = cookies.get("my_stocks")
    if raw_stocks:
        try:
            st.session_state.my_stocks = json.loads(raw_stocks)
        except: pass
    
    st.session_state.tg_token = cookies.get("tg_token") or ""
    st.session_state.tg_chat_id = cookies.get("tg_chat_id") or ""
    st.session_state.tg_threshold = float(cookies.get("tg_threshold") or 3.0)
    st.session_state.cookie_loaded = True

def save_all_to_cookie():
    cookies["my_stocks"] = json.dumps(st.session_state.my_stocks, ensure_ascii=False)
    cookies["tg_token"] = st.session_state.tg_token
    cookies["tg_chat_id"] = st.session_state.tg_chat_id
    cookies["tg_threshold"] = str(st.session_state.tg_threshold)
    cookies.save()

# --- 1. 分析與決策引擎 ---
@st.cache_data(ttl=300)
def fetch_and_analyze(stock_id):
    df = pd.DataFrame()
    for suffix in [".TW", ".TWO"]:
        try:
            temp_df = yf.download(f"{stock_id}{suffix}", period="6mo", progress=False)
            if not temp_df.empty:
                df = temp_df
                break
        except: continue
    
    if df.empty: return None

    # 處理 yfinance 多層索引
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    df = df.astype(float).ffill()

    close = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)
    high = pd.Series(df['High'].values.flatten(), index=df.index).astype(float)
    low = pd.Series(df['Low'].values.flatten(), index=df.index).astype(float)

    try:
        df['MA5'] = SMAIndicator(close, window=5).sma_indicator()
        df['MA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['MA20'] = SMAIndicator(close, window=20).sma_indicator()
        stoch = StochasticOscillator(high, low, close, window=9, window_context=3)
        df['K'] = stoch.stoch(); df['D'] = stoch.stoch_signal()
        df['MACD_diff'] = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df['BBM'] = BollingerBands(close, window=20).bollinger_mavg()
    except: return None

    last = df.iloc[-1]; prev = df.iloc[-2]
    score = 0
    details = []

    # 均線
    if last['MA5'] > last['MA10'] > last['MA20']:
        details.append("✅ 均線多頭"); score += 1

    # KD 進階邏輯 (嚴格交叉 + 區間過濾)
    is_gold_cross = prev['K'] <= prev['D'] and last['K'] > last['D']
    if is_gold_cross:
        if last['K'] < 25:
            details.append("🌟 低檔金叉"); score += 1
        elif 50 <= last['K'] < 80:
            details.append("✅ 轉強金叉"); score += 1
    
    if (df['K'].iloc[-3:] < 20).all():
        details.append("⚠️ 低檔鈍化")

    # 其他指標
    if last['MACD_diff'] > 0: details.append("✅ MACD轉正"); score += 1
    if last['RSI'] > 50: details.append("✅ RSI強勢"); score += 1
    if last['Close'] > last['BBM']: details.append("✅ 站上月線"); score += 1

    decision_map = {
        5: {"grade": "S (極強)", "action": "🔥 續抱/加碼", "color": "red"},
        4: {"grade": "A (強勢)", "action": "🚀 偏多持股", "color": "orange"},
        3: {"grade": "B (轉強)", "action": "📈 少量試單", "color": "green"},
        2: {"grade": "C (盤整)", "action": "⚖️ 暫時觀望", "color": "blue"},
        1: {"grade": "D (弱勢)", "action": "📉 減碼避險", "color": "gray"},
        0: {"grade": "E (極弱)", "action": "🚫 觀望不進場", "color": "black"}
    }
    f_score = min(max(int(score), 0), 5)
    res = decision_map.get(f_score)

    return {
        "price": float(last['Close']),
        "pct": (float(last['Close'])-float(prev['Close']))/float(prev['Close'])*100,
        "grade": res["grade"], "action": res["action"], "color": res["color"],
        "details": details, "score": f_score
    }

# --- 2. 介面 ---
st.title("🤖 台股 AI 技術分級決策支援")

with st.container(border=True):
    st.subheader("🔍 新增自選股票")
    c1, c2, c3 = st.columns([2,3,1.2])
    input_id = c1.text_input("代號", key="add_id")
    input_name = c2.text_input("名稱", key="add_name")
    if c3.button("➕ 新增", use_container_width=True):
        if input_id and input_name:
            if not any(s['id'] == input_id for s in st.session_state.my_stocks):
                st.session_state.my_stocks.append({"id": input_id, "name": input_name})
                save_all_to_cookie()
                st.rerun()

with st.sidebar:
    st.header("⚙️ 通知設定")
    st.session_state.tg_token = st.text_input("Bot Token", type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.tg_chat_id)
    st.session_state.tg_threshold = st.number_input("通知門檻 (%)", value=st.session_state.tg_threshold)
    if st.button("💾 儲存設定"):
        save_all_to_cookie()
        st.success("設定已儲存！")

# --- 3. 顯示股票卡片 ---
st.divider()
# 這裡改用一個容器來承載所有卡片，確保渲染穩定
stock_container = st.container()

with stock_container:
    # 複製一份清單以免在迴圈中刪除導致錯誤
    for idx, stock in enumerate(list(st.session_state.my_stocks)):
        res = fetch_and_analyze(stock['id'])
        
        with st.container(border=True):
            if res:
                col_info, col_metric, col_del = st.columns([3, 2, 0.6])
                with col_info:
                    st.write(f"### {stock['name']} ({stock['id']})")
                    st.markdown(f"評級：`{res['grade']}`")
                    st.markdown(f"**決策：<span style='color:{res['color']}'>{res['action']}</span>**", unsafe_allow_html=True)
                    st.caption(" | ".join(res['details']) if res['details'] else "無顯著訊號")
                with col_metric:
                    st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
                with col_del:
                    if st.button("🗑️", key=f"del_{stock['id']}"):
                        st.session_state.my_stocks.pop(idx)
                        save_all_to_cookie()
                        st.rerun()
            else:
                st.warning(f"無法取得 {stock['name']} ({stock['id']}) 資料")
                if st.button("🗑️ 刪除無效標的", key=f"del_err_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_all_to_cookie()
                    st.rerun()

if st.button("🔄 全部重新整理"):
    st.cache_data.clear()
    st.rerun()
