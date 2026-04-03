
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
from datetime import datetime, time as dt_time
from FinMind.data import DataLoader
from streamlit_cookies_manager import CookieManager
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# --- 0. 基礎設定 ---
st.set_page_config(page_title="台股 AI 決策系統 V7.6", layout="centered")
tw_tz = pytz.timezone('Asia/Taipei')
fm_loader = DataLoader()

# 初始化 session_state
if 'my_stocks' not in st.session_state:
    st.session_state.my_stocks = [{"id": "2330", "name": "台積電"}]
if 'tg_token' not in st.session_state:
    st.session_state.tg_token = ""
if 'tg_chat_id' not in st.session_state:
    st.session_state.tg_chat_id = ""

# Cookie 管理
cookies = CookieManager(prefix="twstock_")
if not cookies.ready():
    st.info("系統初始化中...")
    st.stop()

# 載入 Cookie
if 'cookie_loaded' not in st.session_state:
    raw = cookies.get("my_stocks")
    if raw:
        try: st.session_state.my_stocks = json.loads(raw)
        except: pass
    st.session_state.tg_token = cookies.get("tg_token") or ""
    st.session_state.tg_chat_id = cookies.get("tg_chat_id") or ""
    st.session_state.cookie_loaded = True

def save_all_to_cookie():
    cookies["my_stocks"] = json.dumps(st.session_state.my_stocks, ensure_ascii=False)
    cookies["tg_token"] = st.session_state.tg_token
    cookies["tg_chat_id"] = st.session_state.tg_chat_id
    cookies.save()

# --- 1. 分析引擎 ---
@st.cache_data(ttl=60) # 盤中建議快取時間縮短為 60 秒
def fetch_and_analyze(stock_id):
    now_tw = datetime.now(tw_tz)
    # 判斷是否為台股交易時間 (週一至週五 09:00 - 13:35)
    is_trading = now_tw.weekday() < 5 and dt_time(9, 0) <= now_tw.time() <= dt_time(13, 35)

    df = pd.DataFrame()
    for suffix in [".TW", ".TWO"]:
        try:
            # 盤中抓取 1d (包含今天跳動價), 盤後抓取較長區間確保準確
            temp_df = yf.download(f"{stock_id}{suffix}", period="6mo", progress=False, auto_adjust=True)
            if not temp_df.empty and len(temp_df) > 20:
                df = temp_df
                break
        except: continue
    
    if df.empty: return None

    # 統一格式處理
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).capitalize() for col in df.columns]
    df = df.ffill().astype(float)

    try:
        close = pd.Series(df['Close'].values.flatten(), index=df.index)
        high = pd.Series(df['High'].values.flatten(), index=df.index)
        low = pd.Series(df['Low'].values.flatten(), index=df.index)

        # 指標計算
        df['MA5'] = SMAIndicator(close, window=5).sma_indicator()
        df['MA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['MA20'] = SMAIndicator(close, window=20).sma_indicator()
        
        stoch = StochasticOscillator(high, low, close, window=9, window_context=3)
        df['K'] = stoch.stoch(); df['D'] = stoch.stoch_signal()
        df['MACD_diff'] = MACD(close).macd_diff()
        df['RSI'] = RSIIndicator(close).rsi()
        df['BBM'] = BollingerBands(close).bollinger_mavg()
        
        # 取得最後兩筆資料
        last = df.iloc[-1]; prev = df.iloc[-2]
        score = 0
        details = []

        # 1. 均線排列
        if last['MA5'] > last['MA10'] > last['MA20']:
            details.append("✅ 均線多頭"); score += 1

        # 2. 進階 KD 判斷 (嚴格金叉)
        is_gold_cross = prev['K'] <= prev['D'] and last['K'] > last['D']
        if is_gold_cross:
            if last['K'] < 25:
                details.append("🌟 低檔強勢金叉"); score += 1
            elif 50 <= last['K'] < 80:
                details.append("✅ 轉強黃金交叉"); score += 1
        
        # 3. 鈍化偵測
        if (df['K'].iloc[-3:] < 20).all():
            details.append("⚠️ 低檔鈍化")

        # 4. 其他指標
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
        res = decision_map.get(min(max(int(score), 0), 5))
        
        return {
            "price": float(last['Close']),
            "pct": (float(last['Close'])-float(prev['Close']))/float(prev['Close'])*100,
            "grade": res["grade"], "action": res["action"], "color": res["color"],
            "details": details,
            "is_live": is_trading  # 標記是否為盤中即時資料
        }
    except: return None

# --- 2. 介面 ---
st.title("🤖 台股 AI 決策系統 (盤中即時整合版)")

with st.container(border=True):
    st.subheader("🔍 新增自選股票")
    c1, c2, c3 = st.columns([2,3,1.2])
    input_id = c1.text_input("代號", key="add_id", placeholder="2330")
    input_name = c2.text_input("名稱 (選填)", key="add_name")
    
    if c3.button("➕ 新增", use_container_width=True):
        if input_id:
            final_name = input_name
            if not final_name:
                try:
                    info = fm_loader.taiwan_stock_info()
                    final_name = info[info['stock_id'] == input_id]['stock_name'].values[0]
                except: final_name = f"股票 {input_id}"
            if not any(s['id'] == input_id for s in st.session_state.my_stocks):
                st.session_state.my_stocks.append({"id": input_id, "name": final_name})
                save_all_to_cookie(); st.rerun()

with st.sidebar:
    st.header("⚙️ 系統狀態")
    now_time = datetime.now(tw_tz).strftime("%H:%M:%S")
    st.write(f"目前時間：{now_time}")
    if st.button("🔄 手動刷新資料"):
        st.cache_data.clear(); st.rerun()

# --- 3. 顯示卡片 ---
st.divider()
for idx, stock in enumerate(list(st.session_state.my_stocks)):
    res = fetch_and_analyze(stock['id'])
    with st.container(border=True):
        if res:
            col1, col2, col3 = st.columns([3, 2, 0.6])
            with col1:
                live_tag = "🔴 盤中" if res.get("is_live") else "⚪ 盤後"
                st.write(f"### {stock['name']} ({stock['id']}) {live_tag}")
                st.markdown(f"評級：`{res['grade']}`")
                st.markdown(f"**決策：<span style='color:{res['color']}'>{res['action']}</span>**", unsafe_allow_html=True)
                st.caption(" | ".join(res['details']) if res['details'] else "無顯著訊號")
            with col2:
                st.metric("目前價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col3:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_all_to_cookie(); st.rerun()
        else:
            st.error(f"❌ {stock['id']} 資料讀取失敗")
            if st.button("🗑️ 刪除", key=f"err_{stock['id']}"):
                st.session_state.my_stocks.pop(idx)
                save_all_to_cookie(); st.rerun()
