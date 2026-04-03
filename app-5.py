
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
tw_tz = pytz.timezone('Asia/Taipei')

# Cookie 管理：確保自選清單跨瀏覽器保存
cookies = CookieManager(prefix="twstock_")
if not cookies.ready():
    st.info("正在初始化系統並載入清單，請稍候...")
    st.stop()

def load_stocks_from_cookie():
    raw = cookies.get("my_stocks")
    if raw:
        try:
            return json.loads(raw)
        except:
            pass
    return [{"id": "2330", "name": "台積電"}]

def save_stocks_to_cookie(stocks):
    cookies["my_stocks"] = json.dumps(stocks, ensure_ascii=False)
    cookies.save()

def load_tg_from_cookie():
    return {
        "tg_token": cookies.get("tg_token") or "",
        "tg_chat_id": cookies.get("tg_chat_id") or "",
        "tg_threshold": float(cookies.get("tg_threshold") or 3.0),
    }

def save_tg_to_cookie():
    cookies["tg_token"] = st.session_state.tg_token
    cookies["tg_chat_id"] = st.session_state.tg_chat_id
    cookies["tg_threshold"] = str(st.session_state.tg_threshold)
    cookies.save()

# 初始化 session_state
if 'initialized' not in st.session_state:
    tg = load_tg_from_cookie()
    st.session_state.update({
        'my_stocks': load_stocks_from_cookie(),
        'tg_token': tg["tg_token"],
        'tg_chat_id': tg["tg_chat_id"],
        'tg_threshold': tg["tg_threshold"],
        'initialized': True
    })

# --- 1. 分析與決策引擎 ---
@st.cache_data(ttl=300) # 快取5分鐘，避免頻繁請求被 Yahoo 封鎖
def fetch_and_analyze(stock_id):
    df = pd.DataFrame()
    # 嘗試 台灣上市(.TW) 與 上櫃(.TWO) 標籤
    for suffix in [".TW", ".TWO"]:
        try:
            temp_df = yf.download(f"{stock_id}{suffix}", period="6mo", progress=False)
            if not temp_df.empty:
                df = temp_df
                break
        except: continue
    
    if df.empty: return None

    # 關鍵：處理 yfinance 新版的多層索引問題，避免抓不到 Close
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    df = df.astype(float).ffill()

    # 轉換 Series 確保指標計算穩定
    close = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)
    high = pd.Series(df['High'].values.flatten(), index=df.index).astype(float)
    low = pd.Series(df['Low'].values.flatten(), index=df.index).astype(float)

    try:
        # 計算指標
        df['MA5'] = SMAIndicator(close, window=5).sma_indicator()
        df['MA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['MA20'] = SMAIndicator(close, window=20).sma_indicator()
        
        # 9日 KD
        stoch = StochasticOscillator(high, low, close, window=9, window_context=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        
        df['MACD_diff'] = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df['BBM'] = BollingerBands(close, window=20).bollinger_mavg()
    except: return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    details = []

    # 判斷 A：均線多頭
    if last['MA5'] > last['MA10'] > last['MA20']:
        details.append("✅ 均線多頭排列"); score += 1

    # 判斷 B：新版進階 KD 邏輯
    # 1. 嚴格交叉：昨天 K<=D 且 今天 K>D
    is_gold_cross = prev['K'] <= prev['D'] and last['K'] > last['D']
    
    # 2. 區間過濾與計分
    if is_gold_cross:
        if last['K'] < 25:
            details.append("🌟 低檔強勢金叉 (20以下)"); score += 1
        elif 50 <= last['K'] < 80:
            details.append("✅ 轉強黃金交叉 (50以上)"); score += 1
        # 註：25-50 之間或 80 以上之交叉不計分

    # 3. 鈍化偵測 (警告性質)
    low_blunt = (df['K'].iloc[-3:] < 20).all()
    if low_blunt:
        details.append("⚠️ 低檔鈍化警告")

    # 判斷 C：MACD 轉正
    if last['MACD_diff'] > 0:
        details.append("✅ MACD 柱狀體轉正"); score += 1
    
    # 判斷 D：RSI 強勢
    if last['RSI'] > 50:
        details.append("✅ RSI 強勢區"); score += 1
    
    # 判斷 E：站穩月線
    if last['Close'] > last['BBM']:
        details.append("✅ 站穩月線(MA20)"); score += 1

    # 等級映射表
    decision_map = {
        5: {"grade": "S (極強)", "action": "🔥 續抱/加碼", "color": "red"},
        4: {"grade": "A (強勢)", "action": "🚀 偏多持股", "color": "orange"},
        3: {"grade": "B (轉強)", "action": "📈 少量試單", "color": "green"},
        2: {"grade": "C (盤整)", "action": "⚖️ 暫時觀望", "color": "blue"},
        1: {"grade": "D (弱勢)", "action": "📉 減碼避險", "color": "gray"},
        0: {"grade": "E (極弱)", "action": "🚫 觀望不進場", "color": "black"}
    }
    
    final_score = min(max(int(score), 0), 5)
    res = decision_map.get(final_score)

    return {
        "price": float(last['Close']),
        "pct": (float(last['Close'])-float(prev['Close']))/float(prev['Close'])*100,
        "grade": res["grade"],
        "action": res["action"],
        "color": res["color"],
        "details": details,
        "score": final_score
    }

# --- 2. 介面設計 ---
st.set_page_config(page_title="台股決策系統 V7.2", layout="centered")
st.title("🤖 台股 AI 技術分級決策支援")

# 新增股票區
with st.container(border=True):
    st.subheader("🔍 新增自選股票")
    c1, c2, c3 = st.columns([2,3,1.2])
    input_id = c1.text_input("代號", key="add_id", placeholder="例如: 2330")
    input_name = c2.text_input("名稱", key="add_name", placeholder="例如: 台積電")
    if c3.button("➕ 新增", use_container_width=True):
        if input_id and input_name:
            if not any(s['id'] == input_id for s in st.session_state.my_stocks):
                st.session_state.my_stocks.append({"id": input_id, "name": input_name})
                save_stocks_to_cookie(st.session_state.my_stocks)
                st.rerun()

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 通知設定")
    st.session_state.tg_token = st.text_input("Bot Token", type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.tg_chat_id)
    st.session_state.tg_threshold = st.number_input("通知門檻 (%)", value=st.session_state.tg_threshold)
    if st.button("💾 儲存並刷新"):
        save_tg_to_cookie()
        st.cache_data.clear(); st.rerun()
    st.divider()

# --- 3. 渲染股票卡片 ---
st.divider()
for idx, stock in enumerate(st.session_state.my_stocks):
    res = fetch_and_analyze(stock['id'])
    
    with st.container(border=True):
        if res:
            col_info, col_metric, col_del = st.columns([3, 2, 0.6])
            with col_info:
                st.write(f"### {stock['name']} ({stock['id']})")
                st.markdown(f"評級：`{res['grade']}`")
                st.markdown(f"**建議決策：<span style='color:{res['color']}'>{res['action']}</span>**", unsafe_allow_html=True)
                indicators = " | ".join(res['details']) if res['details'] else "無顯著訊號"
                st.markdown(f"<small>{indicators}</small>", unsafe_allow_html=True)
            with col_metric:
                st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col_del:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_stocks_to_cookie(st.session_state.my_stocks)
                    st.rerun()
        else:
            # 渲染失敗時的防錯處理
            c1, c2 = st.columns([5, 1])
            c1.warning(f"⚠️ 無法取得 {stock['name']} ({stock['id']}) 資料，可能為代號錯誤或 Yahoo 暫時斷線。")
            if c2.button("🗑️", key=f"del_err_{stock['id']}"):
                st.session_state.my_stocks.pop(idx)
                save_stocks_to_cookie(st.session_state.my_stocks)
                st.rerun()

if st.button("🔄 全部重新整理"):
    st.cache_data.clear(); st.rerun()
