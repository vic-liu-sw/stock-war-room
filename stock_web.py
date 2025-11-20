
# 設定頁面標題
st.set_page_config(page_title="全球金融戰情室", layout="wide")

# === 加入這段 CSS 代碼來隱藏右上角選單與 footer ===
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# =================================================


import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 設定頁面標題
st.set_page_config(page_title="全球金融戰情室", layout="wide")
st.title("💹 全球金融戰情室 (ATR 水平支撐壓力版)")

# ==========================================
#  核心邏輯 (處理代碼)
# ==========================================
def process_ticker(code):
    code = code.strip().upper()
    # 處理常見的特殊代碼對應到 Yahoo Finance 格式
    if code in ['USDIDX', 'DXY']: return 'DX-Y.NYB' # 美元指數
    if code in ['GOLD']: return 'GC=F' # 黃金期貨
    if code == 'BITCOIN': return 'BTC-USD'
    
    # 簡單判斷台股 (純數字且長度大於3，預設加上 .TW)
    if code.isdigit() and len(code) >= 3: return f"{code}.TW"
    return code

# ==========================================
#  側邊欄設定
# ==========================================
with st.sidebar:
    st.header("參數設定")

    # --- 熱門代碼字典 ---
    popular_tickers = {
        "自訂輸入 (Manual Input)": "CUSTOM",
        "黃金期貨 (Gold)": "GC=F",      # <--- 已改為直接使用期貨代碼
        "比特幣 (BTC)": "BTC-USD",
        "以太幣 (ETH)": "ETH-USD",
        "台積電 (2330)": "2330.TW",
        "輝達 (Nvidia)": "NVDA",
        "特斯拉 (Tesla)": "TSLA",
        "納斯達克指數 (Nasdaq)": "^IXIC",
        "美元指數 (DXY)": "DX-Y.NYB",
        "蘋果 (Apple)": "AAPL",
        "微軟 (Microsoft)": "MSFT"
    }

    # 使用下拉選單
    selected_label = st.selectbox("🎯 選擇熱門商品", list(popular_tickers.keys()))
    
    # 邏輯判斷
    if popular_tickers[selected_label] == "CUSTOM":
        raw_code = st.text_input("輸入代碼 (例如: 2330, GC=F, NVDA)", value="2330")
        stock_code = raw_code
    else:
        stock_code = popular_tickers[selected_label]
        st.info(f"已選擇代碼: {stock_code}")

    st.markdown("---")
    st.subheader("圖表設定")
    time_frame = st.selectbox("📊 K線週期", ["日線", "週線", "月線"], index=0)
    display_count = st.number_input("顯示 K 棒數量", value=120, step=20)
    
    st.markdown("---")
    st.subheader("指標設定")
    show_hlines = st.checkbox("顯示 ATR 水平支撐/壓力線", value=True)
    show_bb = st.checkbox("顯示布林通道", value=False)
    show_kd = st.checkbox("顯示 KD", value=True)
    show_atr = st.checkbox("顯示 ATR", value=True)
    ma_choices = st.multiselect("均線 (MA)", [5, 10, 20, 60, 120, 240], default=[20, 60])

    st.markdown("---")
    st.subheader("多檔比較")
    comp_a = st.text_input("比較 A", value=stock_code)
    comp_b = st.text_input("比較 B", value="^TWII")

# ==========================================
#  計算邏輯函數
# ==========================================
def calculate_kd(df, period=9):
    low_list = df['Low'].rolling(window=period).min()
    high_list = df['High'].rolling(window=period).max()
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr_list = pd.concat([(high-low), (high-close_prev).abs(), (low-close_prev).abs()], axis=1)
    tr = tr_list.max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_bbands(df, period=20, std_dev=2):
    m = df['Close'].rolling(window=period).mean()
    s = df['Close'].rolling(window=period).std()
    return m + (s * std_dev), m, m - (s * std_dev)

def resample_data(df, freq_str):
    logic = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
    if freq_str == "週線": return df.resample('W-FRI').agg(logic).dropna()
    elif freq_str == "月線":
        try: return df.resample('ME').agg(logic).dropna()
        except: return df.resample('M').agg(logic).dropna()
    return df

# ==========================================
#  主畫面
# ==========================================
tab1, tab2 = st.tabs(["📊 技術分析", "⚖️ 比較"])

with tab1:
    if st.button("開始分析", type="primary"):
        ticker = process_ticker(stock_code)
        
        days_map = {"日線": 5, "週線": 30, "月線": 100}
        fetch_days = max(display_count * days_map[time_frame], 800)
        start_date = datetime.now() - timedelta(days=fetch_days)
        
        with st.spinner(f"正在計算 {ticker} ({time_frame})..."):
            try:
                # 下載數據
                raw_df = yf.download(ticker, start=start_date, progress=False)
                
                # 處理 MultiIndex
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                
                # 檢查是否為空 (針對美元指數保留救援機制)
                if raw_df.empty:
                    if ticker == 'DX-Y.NYB':
                        st.warning(f"⚠️ Yahoo 美元指數 (DX-Y.NYB) 暫時無法讀取，已自動切換至「美元指數期貨 (DX=F)」以供分析。")
                        ticker = 'DX=F'
                        raw_df = yf.download(ticker, start=start_date, progress=False)
                    
                    # 再次處理 MultiIndex
                    if isinstance(raw_df.columns, pd.MultiIndex):
                        raw_df.columns = raw_df.columns.get_level_values(0)
                
                # 轉型與清理
                for c in ['Open','High','Low','Close','Volume']:
                    if c in raw_df.columns:
                        raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
                
                raw_df.dropna(inplace=True)

                if raw_df.empty:
                    st.error(f"查無資料: {ticker}。")
                else:
                    # 轉換週期
                    plot_df = resample_data(raw_df, time_frame)
                    plot_df['ATR'] = calculate_atr(plot_df)
                    
                    if len(plot_df) < 2:
                        st.warning("資料不足以計算指標")
                    else:
                        last_bar = plot_df.iloc[-1]
                        prev_bar = plot_df.iloc[-2]
                        cur_atr = last_bar['ATR']
                        
                        res_level = last_bar['High'] + cur_atr
                        sup_level = last_bar['Low'] - cur_atr
                        
                        is_up = last_bar['Close'] >= prev_bar['Close']
                        change = last_bar['Close'] - prev_bar['Close']
                        change_pct = (change / prev_bar['Close']) * 100
                        period_name = "今日" if time_frame == "日線" else "本週" if time_frame == "週線" else "本月"
                        
                        st.markdown("### 📈 策略數據看板")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(f"{period_name}開盤", f"{last_bar['Open']:.2f}")
                        c2.metric("最新收盤", f"{last_bar['Close']:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
                        c3.metric(f"壓力 (綠線)", f"{res_level:.2f}")
                        c4.metric(f"支撐 (紅線)", f"{sup_level:.2f}")

                        # 繪圖
                        chart_data = plot_df.iloc[-display_count:].copy()
                        add_plots = []
                        pidx = 2
                        
                        h_lines_dict = None
                        if show_hlines:
                            h_lines_dict = dict(
                                hlines=[res_level, sup_level],
                                colors=['green', 'red'],
                                linestyle='-', linewidths=1.5, alpha=0.8
                            )

                        if show_bb:
                            u, m, l = calculate_bbands(chart_data)
                            add_plots.append(mpf.make_addplot(u, panel=0, color='gray', linestyle='--', width=0.6))
                            add_plots.append(mpf.make_addplot(l, panel=0, color='gray', linestyle='--', width=0.6))

                        if show_atr:
                            add_plots.append(mpf.make_addplot(chart_data['ATR'], panel=pidx, color='#00bcd4', title='ATR'))
                            pidx += 1

                        if show_kd:
                            k, d = calculate_kd(chart_data)
                            add_plots.append(mpf.make_addplot([80]*len(chart_data), panel=pidx, color='gray', linestyle=':', width=0.8))
                            add_plots.append(mpf.make_addplot([20]*len(chart_data), panel=pidx, color='gray', linestyle=':', width=0.8))
                            add_plots.append(mpf.make_addplot(k, panel=pidx, color='orange', title='KD'))
                            add_plots.append(mpf.make_addplot(d, panel=pidx, color='purple'))
                            pidx += 1

                        # 配色：台股紅漲綠跌，外匯/美股綠漲紅跌
                        if '.TW' in ticker:
                            mc = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit', volume='inherit')
                        else:
                            mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='inherit')
                            
                        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
                        ratios = [3, 1] + [1] * (pidx - 2)

                        fig, ax = mpf.plot(
                            chart_data, type='candle', mav=tuple(ma_choices) if ma_choices else None,
                            volume=True, addplot=add_plots, style=s, returnfig=True,
                            title=f"{ticker} - {time_frame} (ATR Level)", figsize=(12, 10),
                            panel_ratios=ratios, hlines=h_lines_dict
                        )
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"發生錯誤: {e}")

# --- 分頁 2: 比較 ---
with tab2:
    if st.button("執行比較", type="primary"):
        ca, cb = process_ticker(comp_a), process_ticker(comp_b)
        s_date = datetime.now() - timedelta(days=365)
        
        def get_d(t):
            try:
                d = yf.download(t, start=s_date, progress=False)
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                c = 'Adj Close' if 'Adj Close' in d.columns else 'Close'
                return pd.to_numeric(d[c], errors='coerce').dropna()
            except: return None

        with st.spinner("比較中..."):
            da, db = get_d(ca), get_d(cb)
            if da is not None and db is not None:
                df = pd.concat([da, db], axis=1).dropna()
                df.columns = [ca, cb]
                
                if not df.empty:
                    st.markdown(f"### {ca} vs {cb} (近一年走勢)")
                    st.metric("相關係數", f"{df[ca].corr(df[cb]):.4f}")
                    st.line_chart(df / df.iloc[0] * 100)
                else:
                    st.error("合併後無重疊資料")
            else:
                st.error("資料讀取失敗")
