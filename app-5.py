import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import pytz
import json
from datetime import datetime, time as dt_time
from streamlit_cookies_manager import CookieManager
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# --- 0. 基礎設定 ---
tw_tz = pytz.timezone('Asia/Taipei')

# Cookie 管理
cookies = CookieManager(prefix="twstock_")
if not cookies.ready():
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

if 'initialized' not in st.session_state:
    tg = load_tg_from_cookie()
    st.session_state.update({
        'my_stocks': load_stocks_from_cookie(),
        'tg_token': tg["tg_token"],
        'tg_chat_id': tg["tg_chat_id"],
        'tg_threshold': tg["tg_threshold"],
        'initialized': True,
        'alert_history': {}
    })

# --- 1. 分析與決策引擎 ---
@st.cache_data(ttl=60)
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
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.astype(float).ffill()

    close = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)
    high = pd.Series(df['High'].values.flatten(), index=df.index).astype(float)
    low = pd.Series(df['Low'].values.flatten(), index=df.index).astype(float)

    try:
        # 計算技術指標
        df['MA5'] = SMAIndicator(close, window=5).sma_indicator()
        df['MA10'] = SMAIndicator(close, window=10).sma_indicator()
        df['MA20'] = SMAIndicator(close, window=20).sma_indicator()
        stoch = StochasticOscillator(high, low, close, window=9, window_context=3)
        df['K']=stoch.stoch(); df['D']=stoch.stoch_signal()
        df['MACD_diff'] = MACD(close, window_slow=26, window_fast=12, window_sign=9).macd_diff()
        df['RSI'] = RSIIndicator(close, window=14).rsi()
        df['BBM'] = BollingerBands(close, window=20).bollinger_mavg()
    except: return None

    last = df.iloc[-1]; prev = df.iloc[-2]
    score = 0
    details = []

    # A. 均線判斷
    if last['MA5'] > last['MA10'] > last['MA20']:
        details.append("✅ 均線多頭排列"); score += 1

    # B. 進階 KD 判斷邏輯 (依據使用者直覺優化)
    # 1. 必須是今日才發生的「黃金交叉」
    is_gold_cross = prev['K'] <= prev['D'] and last['K'] > last['D']
    # 2. 篩選優質交叉區間 (25以下低檔 或 50以上轉強) 且 排除 80 以上高檔過熱
    if is_gold_cross and last['K'] < 80:
        if last['K'] < 25:
            details.append("🌟 低檔強勢金叉"); score += 1
        elif last['K'] > 50:
            details.append("✅ 轉強黃金交叉"); score += 1
    
    # 3. 額外偵測：低檔鈍化警告 (不加分，僅提醒)
    low_blunt = (df['K'].iloc[-3:] < 20).all()
    if low_blunt:
        details.append("⚠️ 低檔鈍化警告")

    # C. MACD 判斷
    if last['MACD_diff'] > 0:
        details.append("✅ MACD 柱狀體轉正"); score += 1
    
    # D. RSI 判斷
    if last['RSI'] > 50:
        details.append("✅ RSI 強勢區"); score += 1
    
    # E. 價格位置判斷
    if last['Close'] > last['BBM']:
        details.append("✅ 站穩月線(MA20)"); score += 1

    # 決策對照表 (維持原等級)
    decision_map = {
        5: {"grade": "S (極強)", "action": "🔥 續抱/加碼", "color": "red"},
        4: {"grade": "A (強勢)", "action": "🚀 偏多持股", "color": "orange"},
        3: {"grade": "B (轉強)", "action": "📈 少量試單", "color": "green"},
        2: {"grade": "C (盤整)", "action": "⚖️ 暫時觀望", "color": "blue"},
        1: {"grade": "D (弱勢)", "action": "📉 減碼避險", "color": "gray"},
        0: {"grade": "E (極弱)", "action": "🚫 觀望不進場", "color": "black"}
    }
    
    # 確保 score 不會超過 5
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

# --- 2. 介面 ---
st.set_page_config(page_title="台股決策系統 V7.2", layout="centered")
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
                save_stocks_to_cookie(st.session_state.my_stocks)
                st.rerun()

with st.sidebar:
    st.header("⚙️ 通知設定")
    st.session_state.tg_token = st.text_input("Bot Token", type="password", value=st.session_state.tg_token)
    st.session_state.tg_chat_id = st.text_input("Chat ID", value=st.session_state.tg_chat_id)
    st.session_state.tg_threshold = st.number_input("通知門檻 (%)", value=st.session_state.tg_threshold)
    if st.button("💾 儲存並刷新"):
        save_tg_to_cookie()
        st.cache_data.clear(); st.rerun()
    st.divider()

    if st.button("🚀 手動測試掃描並發送通知", use_container_width=True):
        st.cache_data.clear()
        found = 0
        for s in st.session_state.my_stocks:
            res = fetch_and_analyze(s['id'])
            if res and abs(res['pct']) >= st.session_state.tg_threshold:
                msg = (f"🔔 <b>【AI 決策通知】</b>\n\n"
                       f"標的：<b>{s['name']} ({s['id']})</b>\n"
                       f"目前股價：<b>{res['price']:.2f}</b>\n"
                       f"今日漲跌：<b>{res['pct']:+.2f}%</b>\n"
                       f"技術評級：{res['grade']}\n"
                       f"建議決策：<b>{res['action']}</b>\n\n"
                       f"符合指標：{', '.join(res['details']) if res['details'] else '無'}")

                url = f"https://api.telegram.org/bot{st.session_state.tg_token}/sendMessage"
                requests.post(url, json={"chat_id": st.session_state.tg_chat_id, "text": msg, "parse_mode": "HTML"})
                found += 1
        st.success(f"掃描完成，已發送 {found} 則通知")

# --- 3. 顯示清單 ---
st.divider()
for idx, stock in enumerate(st.session_state.my_stocks):
    res = fetch_and_analyze(stock['id'])
    if res:
        with st.container(border=True):
            col_info, col_metric, col_del = st.columns([3, 2, 0.6])
            with col_info:
                st.write(f"### {stock['name']} ({stock['id']})")
                st.markdown(f"評級：`{res['grade']}`")
                st.markdown(f"**建議決策：<span style='color:{res['color']}'>{res['action']}</span>**", unsafe_allow_html=True)
                indicators = "　".join(res['details']) if res['details'] else "無"
                st.markdown(f"符合指標：{indicators}")
            with col_metric:
                st.metric("股價", f"{res['price']:.2f}", f"{res['pct']:+.2f}%", delta_color="inverse")
            with col_del:
                if st.button("🗑️", key=f"del_{stock['id']}"):
                    st.session_state.my_stocks.pop(idx)
                    save_stocks_to_cookie(st.session_state.my_stocks)
                    st.rerun()

if st.button("🔄 全部重新整理"):
    st.cache_data.clear(); st.rerun()
