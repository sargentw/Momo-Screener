import ccxt
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import time

# Load secrets for Bitget API (add in Streamlit Cloud settings)
API_KEY = st.secrets.get('API_KEY', 'YOUR_API_KEY_HERE')  # Fallback for local testing
API_SECRET = st.secrets.get('API_SECRET', 'YOUR_API_SECRET_HERE')

# Initialize Bitget futures exchange with API key
exchange = ccxt.bitget({
    'options': {'defaultType': 'future'},
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True
})

# Load perp USDT futures symbols (limit to 10 for testing; change to [:] for full)
markets = exchange.load_markets()
symbols = [m['symbol'] for m in markets.values() if m.get('perp') and m['quote'] == 'USDT'][:10]

# Function to fetch and compute data for a symbol (4 hours = 240 x 1m candles)
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_symbol_data(symbol, num_1m_candles_4h=240, num_1m_candles_1h=60):
    try:
        # Fetch 1m candles for 4 hours
        ohlcv_1m = exchange.fetch_ohlcv(symbol, '1m', limit=num_1m_candles_4h)
        df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')

        # Aggregate recent 4h data
        latest_1m = df_1m.iloc[-1]
        agg_data = {
            'symbol': symbol,
            'price': latest_1m['close'],
            'volume_4h': df_1m['volume'].sum(),
        }

        # Fetch 1m open interest history for 4 hours
        oi_1m = exchange.fetch_open_interest_history(symbol, '1m', limit=num_1m_candles_4h)
        oi_values = [o['open_interest'] for o in oi_1m]
        agg_data['oi_4h'] = oi_values[-1] if oi_values else np.nan

        # Pearson's R for 1m price linear regression (over 4 hours)
        closes = df_1m['close'].values
        x = np.arange(len(closes))
        if len(closes) >= 2:
            slope, intercept, r, p, se = stats.linregress(x, closes)
            agg_data['pearson_r'] = r
        else:
            agg_data['pearson_r'] = np.nan

        # Slopes for volume and OI over past 1h and 4h
        for period_hours, period_key, num_candles in [(1, '1h', num_1m_candles_1h), (4, '4h', num_1m_candles_4h)]:
            volumes = df_1m['volume'].tail(num_candles).values
            oi_period = oi_values[-num_candles:]

            # Volume slope
            if len(volumes) >= 2:
                x_period = np.arange(len(volumes))
                slope_vol, _, _, _, _ = stats.linregress(x_period, volumes)
                agg_data[f'volume_slope_{period_key}'] = slope_vol
            else:
                agg_data[f'volume_slope_{period_key}'] = np.nan

            # OI slope
            if len(oi_period) >= 2:
                x_period = np.arange(len(oi_period))
                slope_oi, _, _, _, _ = stats.linregress(x_period, oi_period)
                agg_data[f'oi_slope_{period_key}'] = slope_oi
            else:
                agg_data[f'oi_slope_{period_key}'] = np.nan

        return agg_data
    except Exception as e:
        st.write(f"Error fetching data for {symbol}: {str(e)}")  # Display error for debugging
        return None  # Skip symbols with errors

# Streamlit UI
st.title('Bitget Perp Futures Screener (1m Candles, Updates on Refresh)')

if st.button('Refresh Data'):
    # Scan and aggregate data with progress bar
    st.subheader('Scanning symbols...')
    progress_bar = st.progress(0)
    data = []
    for i, s in enumerate(symbols):
        data.append(get_symbol_data(s))
        progress_bar.progress((i + 1) / len(symbols))
        time.sleep(0.5)  # Short delay to avoid rate limits
    data = [d for d in data if d]  # Filter None
    df = pd.DataFrame(data)
    st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    if df.empty:
        st.error("No data fetched. Check if API key is correct in secrets or try refreshing. Check logs for details.")
        st.stop()  # Halt execution to prevent errors

    # Filters/Alerts in sidebar
    st.sidebar.header('Filters/Alerts')
    min_pearson_r = st.sidebar.slider('Min Pearson\'s R', -1.0, 1.0, 0.0)
    min_volume_slope_1h = st.sidebar.number_input('Min Volume Slope (1h)', value=0.0)
    min_volume_slope_4h = st.sidebar.number_input('Min Volume Slope (4h)', value=0.0)
    min_oi_slope_1h = st.sidebar.number_input('Min OI Slope (1h)', value=0.0)
    min_oi_slope_4h = st.sidebar.number_input('Min OI Slope (4h)', value=0.0)
    selected_symbols = st.sidebar.multiselect('Select Symbols', df['symbol'].unique(), default=df['symbol'].unique())

    # Apply filters
    filtered_df = df[
        (df['symbol'].isin(selected_symbols)) &
        (df['pearson_r'] >= min_pearson_r) &
        (df['volume_slope_1h'] >= min_volume_slope_1h) &
        (df['volume_slope_4h'] >= min_volume_slope_4h) &
        (df['oi_slope_1h'] >= min_oi_slope_1h) &
        (df['oi_slope_4h'] >= min_oi_slope_4h)
    ]

    # Display table (sortable, alerts as highlighted rows)
    st.subheader('Screener Results')
    if not filtered_df.empty:
        st.dataframe(filtered_df.style.highlight_max(axis=0, subset=['pearson_r', 'volume_slope_1h', 'volume_slope_4h', 'oi_slope_1h', 'oi_slope_4h'], color='lightgreen'))
        st.success(f'{len(filtered_df)} symbols match filters/alerts!')
    else:
        st.warning('No matches.')

    # Expandable charts
    st.subheader('Charts')
    for idx, row in filtered_df.iterrows():
        if st.expander(f"View Chart for {row['symbol']}"):
            # Fetch detailed 1m data for chart (24 hours = 1440 candles)
            ohlcv = exchange.fetch_ohlcv(row['symbol'], '1m', limit=1440)
            df_chart = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'], unit='ms')
            oi = exchange.fetch_open_interest_history(row['symbol'], '1m', limit=1440)
            df_chart['oi'] = [o['open_interest'] for o in oi]

            # Plotly chart: Candlestick + volume/OI
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart['timestamp'], open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name='Price'))
            fig.add_trace(go.Bar(x=df_chart['timestamp'], y=df_chart['volume'], name='Volume', yaxis='y2', opacity=0.5))
            fig.add_trace(go.Scatter(x=df_chart['timestamp'], y=df_chart['oi'], name='OI', yaxis='y3', mode='lines'))
            fig.update_layout(
                title=f"{row['symbol']} Chart",
                yaxis_title='Price',
                yaxis2=dict(title='Volume', overlaying='y', side='right'),
                yaxis3=dict(title='OI', overlaying='y', side='left', anchor='free', position=0.05),
                xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig)
else:
    st.info('Click "Refresh Data" to start scanning.')
