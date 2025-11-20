import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
#  1. 頁面設定
# ==========================================
st.set_page_config(page_title="全球金融戰情室", layout="wide")

# ==========================================
#  2. 修正後的 CSS (只隱藏選單，不隱藏 Header)
# ==========================================
hide_menu_style = """
    <style>
    /* 隱藏右上角的 ... 選單 (包含 View Source) */
    #MainMenu {visibility: hidden;}
    /* 隱藏底部的 Made with Streamlit */
    footer {visibility: hidden;}
    /* ⚠️ 關鍵修正：不再隱藏 header，這樣手機版左上角的側邊欄按鈕才會出現！ */
    /* header {visibility: hidden;} */
    
    /* 手機版面微調 */
    .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("💹 全球金融戰情室")

# ==========================================
#  核心資料處理函數
# ==========================================
def process_ticker(code):
    code = code.strip().upper()
    if code in ['USDIDX', 'DXY']: return 'DX-Y.NYB'
    if code in ['GOLD']: return 'GC=F'
    if code == 'BITCOIN': return 'BTC-USD'
    if code.isdigit() and len(code) >= 3: return f"{code}.TW"
    return code

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
#  3. 將重要選項移至「主畫面頂部」 (方便手機切換)
# ==========================================
popular_tickers = {
    "台積電 (2330)": "2330.TW",
    "黃金期貨 (Gold)": "GC=F",
    "比特幣 (BTC)": "BTC-USD",
    "以太幣 (ETH)": "ETH-USD",
    "輝達 (NVDA)": "NVDA",
    "特斯拉 (TSLA)": "TSLA",
    "納斯達克 (Nasdaq)": "^IXIC",
    "美元指數 (DXY)": "DX-Y.NYB",
    "蘋果 (AAPL)": "AAPL",
    "微軟 (MSFT)": "MSFT",
    "自訂輸入...": "CUSTOM"
}

# 建立兩欄佈局：左邊選商品，右邊選週期
col1, col2 = st.columns([2, 1])

with col1:
    selected_label = st.selectbox("🎯 選擇商品", list(popular_tickers.keys()))
    if popular_tickers[selected_label] == "CUSTOM":
        stock_code = st.text_input("輸入代碼", value="2330")
    else:
        stock_code = popular_tickers[selected_label]

with col2:
    time_frame = st.selectbox("📊 週期", ["日線", "週線", "月線"], index=0)

# ==========================================
#  4. 側邊欄 (只放次要的參數設定)
# ==========================================
with st.sidebar:
    st.header("進階設定")
    st.info("手機版請點左上角箭頭展開")
    
    display_count = st.number_input("K棒數量", value=120, step=20)
    st.markdown("---")
    st.subheader("指標開關")
    show_hlines = st.checkbox("ATR 支撐/壓力", value=True)
    show_bb = st.checkbox("布林通道", value=False)
    show_kd = st.checkbox("KD 指標", value=True)
    show_atr = st.checkbox("ATR 指標", value=False)
    ma_choices = st.multiselect("均線 (MA)", [5, 10, 20, 60, 120], default=[20, 60])

# ==========================================
#  5. 繪圖邏輯 (Plotly)
# ==========================================
ticker = process_ticker(stock_code)
days_map = {"日線": 5, "週線": 30, "月線": 100}
fetch_days = max(display_count * days_map[time_frame], 800)
start_date = datetime.now() - timedelta(days=fetch_days)

try:
    # 手機版不顯示 spinner 以免閃爍，直接計算
    raw_df = yf.download(ticker, start=start_date, progress=False)
    if isinstance(raw_df.columns, pd.MultiIndex): 
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    if raw_df.empty:
        # 救援：若是美元指數失敗，切換期貨
        if ticker == 'DX-Y.NYB':
            ticker = 'DX=F'
            raw_df = yf.download(ticker, start=start_date, progress=False)
            if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)

    for c in ['Open','High','Low','Close','Volume']:
        if c in raw_df.columns: raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
    raw_df.dropna(inplace=True)

    if raw_df.empty:
        st.error("查無資料，請稍後再試")
    else:
        # 數據處理
        plot_df = resample_data(raw_df, time_frame)
        plot_df['ATR'] = calculate_atr(plot_df)
        chart_data = plot_df.iloc[-display_count:].copy()
        
        # 計算最新數據
        last = chart_data.iloc[-1]
        prev = chart_data.iloc[-2]
        cur_atr = last['ATR']
        res_level = last['High'] + cur_atr
        sup_level = last['Low'] - cur_atr
        change = last['Close'] - prev['Close']
        change_pct = (change / prev['Close']) * 100
        
        # === 價格看板 ===
        st.markdown(f"#### {ticker} : {last['Close']:.2f} <span style='color:{'red' if change>=0 else 'green'}'>({change:+.2f} / {change_pct:+.2f}%)</span>", unsafe_allow_html=True)
        
        k1, k2 = st.columns(2)
        k1.metric("壓力 (綠)", f"{res_level:.2f}")
        k2.metric("支撐 (紅)", f"{sup_level:.2f}")

        # === Plotly 繪圖 ===
        rows = 2 
        if show_kd: rows += 1
        if show_atr: rows += 1
        
        row_heights = [0.5] + [0.15] * (rows - 1)
        
        fig = make_subplots(
            rows=rows, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=row_heights[:rows]
        )

        # 1. K線圖
        color_up = 'red' if '.TW' in ticker else 'green'
        color_down = 'green' if '.TW' in ticker else 'red'

        fig.add_trace(go.Candlestick(
            x=chart_data.index,
            open=chart_data['Open'], high=chart_data['High'],
            low=chart_data['Low'], close=chart_data['Close'],
            name='K線',
            increasing_line_color=color_up, decreasing_line_color=color_down
        ), row=1, col=1)

        # 均線
        for ma in ma_choices:
            ma_line = chart_data['Close'].rolling(window=ma).mean()
            fig.add_trace(go.Scatter(x=chart_data.index, y=ma_line, name=f'MA{ma}', line=dict(width=1)), row=1, col=1)

        # 布林通道
        if show_bb:
            u, m, l = calculate_bbands(chart_data)
            fig.add_trace(go.Scatter(x=chart_data.index, y=u, name='BB上', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=l, name='BB下', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)

        # ATR 線
        if show_hlines:
            fig.add_hline(y=res_level, line_dash="dash", line_color="green", annotation_text="壓力", row=1, col=1)
            fig.add_hline(y=sup_level, line_dash="dash", line_color="red", annotation_text="支撐", row=1, col=1)

        # 2. 成交量
        colors_vol = [color_up if c >= o else color_down for c, o in zip(chart_data['Close'], chart_data['Open'])]
        fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'], name='成交量', marker_color=colors_vol), row=2, col=1)

        current_row = 3
        
        # 3. KD
        if show_kd:
            k, d = calculate_kd(chart_data)
            fig.add_trace(go.Scatter(x=chart_data.index, y=k, name='K', line=dict(color='orange')), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=d, name='D', line=dict(color='purple')), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="gray", row=current_row, col=1)
            current_row += 1

        # 4. ATR
        if show_atr:
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['ATR'], name='ATR', line=dict(color='#00bcd4')), row=current_row, col=1)

        fig.update_layout(
            height=800,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
            dragmode='pan'
        )
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"發生錯誤: {e}")
