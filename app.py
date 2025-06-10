import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
from transformer_model import StockTransformer
import plotly.express as px
from plotly.subplots import make_subplots
import ta
import json
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(
    page_title="NSE Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .compact-metrics .stMetric, .compact-metrics .stMarkdown {
        display: inline-block;
        margin-right: 0.7em;
        margin-bottom: 0.1em;
        font-size: 0.85em;
        padding: 0.1em 0.2em;
    }
    .compact-metrics {
        padding: 0.1em 0.2em 0.1em 0.2em !important;
        margin-bottom: 0.2em !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Global Model Parameters ---
seq_length = 30 # Sequence length for transformer input
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50',
            'BB_high', 'BB_low', 'BB_mid', 'EMA12', 'EMA26']

# Stock database
STOCK_DB = {
    'RELIANCE': 'Reliance Industries Ltd',
    'TCS': 'Tata Consultancy Services Ltd',
    'HDFCBANK': 'HDFC Bank Ltd',
    'INFY': 'Infosys Ltd',
    'ICICIBANK': 'ICICI Bank Ltd',
    'HINDUNILVR': 'Hindustan Unilever Ltd',
    'SBIN': 'State Bank of India',
    'BHARTIARTL': 'Bharti Airtel Ltd',
    'KOTAKBANK': 'Kotak Mahindra Bank Ltd',
    'BAJFINANCE': 'Bajaj Finance Ltd',
    'BAJAJFINSV': 'Bajaj Finserv Ltd',
    'ASIANPAINT': 'Asian Paints Ltd',
    'AXISBANK': 'Axis Bank Ltd',
    'MARUTI': 'Maruti Suzuki India Ltd',
    'TITAN': 'Titan Company Ltd',
    'NESTLEIND': 'Nestle India Ltd',
    'ONGC': 'Oil and Natural Gas Corporation Ltd',
    'POWERGRID': 'Power Grid Corporation of India Ltd',
    'NTPC': 'NTPC Ltd',
    'ULTRACEMCO': 'UltraTech Cement Ltd',
    'SUNPHARMA': 'Sun Pharmaceutical Industries Ltd',
    'TECHM': 'Tech Mahindra Ltd',
    'TATAMOTORS': 'Tata Motors Ltd',
    'ADANIPORTS': 'Adani Ports and Special Economic Zone Ltd',
    'JSWSTEEL': 'JSW Steel Ltd',
    'TATASTEEL': 'Tata Steel Ltd',
    'HCLTECH': 'HCL Technologies Ltd',
    'WIPRO': 'Wipro Ltd',
    'DRREDDY': 'Dr. Reddy\'s Laboratories Ltd',
    'BRITANNIA': 'Britannia Industries Ltd',
    'INDUSINDBK': 'IndusInd Bank Ltd',
    'HDFC': 'Housing Development Finance Corporation Ltd',
    'LT': 'Larsen & Toubro Ltd',
    'ITC': 'ITC Ltd',
    'ASIANPAINT': 'Asian Paints Ltd',
    'BHARATFORG': 'Bharat Forge Ltd',
    'GODREJCP': 'Godrej Consumer Products Ltd',
    'EICHERMOT': 'Eicher Motors Ltd',
    'GRASIM': 'Grasim Industries Ltd',
    'IOC': 'Indian Oil Corporation Ltd',
    'M&M': 'Mahindra & Mahindra Ltd',
    'RECLTD': 'REC Ltd',
    'SAIL': 'Steel Authority of India Ltd',
    'UPL': 'UPL Ltd',
    'ZEEL': 'Zee Entertainment Enterprises Ltd',
    'ADANIENT': 'Adani Enterprises Ltd',
    'APOLLOHOSP': 'Apollo Hospitals Enterprise Ltd',
    'DMART': 'Avenue Supermarts Ltd',
    'BAJAJHLDNG': 'Bajaj Holdings & Investment Ltd',
    'BERGEPAINT': 'Berger Paints India Ltd',
    'BIOCON': 'Biocon Ltd',
    'BANDHANBNK': 'Bandhan Bank Ltd',
    'CADILAHC': 'Cadila Healthcare Ltd',
    'DLF': 'DLF Ltd',
    'DABUR': 'Dabur India Ltd',
    'GODREJPROP': 'Godrej Properties Ltd',
    'JUBLFOOD': 'Jubilant Foodworks Ltd',
    'LUPIN': 'Lupin Ltd',
    'MOTHERSUMI': 'Motherson Sumi Systems Ltd',
    'NAUKRI': 'Info Edge (India) Ltd',
    'PEL': 'Piramal Enterprises Ltd',
    'PETRONET': 'Petronet LNG Ltd',
    'PNB': 'Punjab National Bank',
    'PIDILITIND': 'Pidilite Industries Ltd',
    'SIEMENS': 'Siemens Ltd',
    'SRF': 'SRF Ltd',
    'TVSMOTOR': 'TVS Motor Company Ltd',
    'VEDL': 'Vedanta Ltd',
    'YESBANK': 'YES Bank Ltd',
    'ZYDUSLIFE': 'Zydus Lifesciences Ltd'
}

# Training history file
TRAINING_HISTORY_FILE = 'training_history.json'

# Add this helper for info tooltips
INFO = {
    'MAE': ('Mean Absolute Error', 'Measures average prediction error. Lower is better.'),
    'RMSE': ('Root Mean Squared Error', 'Penalizes larger errors more than MAE. Lower is better.'),
    'RSI': ('Relative Strength Index', 'Momentum oscillator: indicates overbought (>70) or oversold (<30) conditions.'),
    'MACD': ('Moving Average Convergence Divergence', 'Trend-following momentum indicator: shows relationship between two EMAs.'),
    'EMA12': ('12-period Exponential Moving Average', 'Short-term trend indicator. Reacts quickly to price changes.'),
    'EMA26': ('26-period Exponential Moving Average', 'Longer-term trend indicator. Smoother than EMA12.'),
}

def info_icon(label):
    full, desc = INFO[label]
    return f'<span style="cursor:pointer; border-bottom:1px dotted #888; font-size:0.9em;" title="{full}: {desc}">ℹ️</span>'

def load_training_history():
    """Load training history from JSON file"""
    if os.path.exists(TRAINING_HISTORY_FILE):
        with open(TRAINING_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_training_history(history):
    """Save training history to JSON file"""
    with open(TRAINING_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def get_stock_options():
    """Get list of stock options for datalist"""
    return [f"{code} - {name}" for code, name in STOCK_DB.items()]

def get_stock_code(selection):
    """Extract stock code from selection"""
    if ' - ' in selection:
        return selection.split(' - ')[0]
    return selection.upper()

def plot_training_metrics(history):
    """Plot training metrics"""
    if not history:
        return None

    # history is a list of dicts
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Training Loss', 'Validation Loss',
                                     'Accuracy', 'Data Points'))

    fig.add_trace(
        go.Scatter(y=[h['train_loss'] for h in history],
                  name='Training Loss',
                  line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(y=[h['val_loss'] for h in history],
                  name='Validation Loss',
                  line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(y=[h['accuracy'] for h in history],
                  name='Accuracy',
                  line=dict(color='green')),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(y=[h['data_points'] for h in history],
                  name='Data Points',
                  line=dict(color='purple')),
        row=2, col=1
    )

    fig.update_layout(height=800, showlegend=True)
    return fig

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    df['BB_mid'] = bollinger.bollinger_mavg()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    
    # Exponential Moving Averages
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    return df

def prepare_data(df):
    """Prepare data for prediction"""
    global seq_length # Access global seq_length
    
    # Minimum rows required for technical indicators (e.g., SMA_50 needs 50)
    # Plus whatever is needed for seq_length. Choose the higher of the two.
    min_required_rows = max(50, seq_length + 1) # Ensure enough data for both indicators and sequences

    # Check if df is already too small for indicators even before adding them
    if len(df) < min_required_rows:
        # st.warning(f"Initial data for preparation ({len(df)} rows) is too short for all technical indicators and model sequence length ({min_required_rows} required). Returning empty data.")
        return pd.DataFrame(), None, None

    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN values (technical indicators might introduce NaNs at the beginning)
    # Store initial length after indicators for more informative warning
    initial_len_after_indicators = len(df)
    df = df.dropna()
    
    # If no data remains after dropping NaNs, return empty dataframe and None scalers
    if df.empty:
        # st.warning(f"No data remains after adding technical indicators and dropping NaNs. Original length before dropna: {initial_len_after_indicators}. Try a longer timeframe.")
        return pd.DataFrame(), None, None

    # Select features
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_features = ['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 
                         'BB_high', 'BB_low', 'BB_mid', 'EMA12', 'EMA26']
    
    # Normalize data
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    price_scaler = RobustScaler()
    indicator_scaler = MinMaxScaler()
    
    print(f"DEBUG: Before scaling price features - df[price_features] shape: {df[price_features].shape}")
    # Ensure there's data to fit the scalers
    if not df[price_features].empty:
        df[price_features] = price_scaler.fit_transform(df[price_features])
    else:
        price_scaler = None # Or handle as a warning/error later

    print(f"DEBUG: Before scaling indicator features - df[indicator_features] shape: {df[indicator_features].shape}")
    if not df[indicator_features].empty:
        df[indicator_features] = indicator_scaler.fit_transform(df[indicator_features])
    else:
        indicator_scaler = None # Or handle as a warning/error later
    
    return df, price_scaler, indicator_scaler

def plot_stock_data(df, predictions=None):
    """Create an interactive plot with Plotly"""
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.6, 0.2, 0.2])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'),
                  row=1, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='BB High', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='BB Low', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_mid'], name='BB Mid', line=dict(color='gray')), row=1, col=1)

    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], 
                                y=predictions,
                                name='Predictions',
                                line=dict(color='red', dash='dot')),
                     row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Stock Analysis',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='RSI',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(
        input_dim=len(features),
        d_model=32,
        nhead=4,
        num_layers=2
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model, device

def calculate_renko(df, brick_size=2):
    """Calculate Renko chart data"""
    # Calculate price changes
    df['price_change'] = df['Close'].diff()
    
    # Initialize Renko data
    renko_data = []
    current_price = df['Close'].iloc[0]
    current_brick = 0
    last_direction = 0  # Track last direction
    
    for i in range(1, len(df)):
        price_change = df['price_change'].iloc[i]
        price = df['Close'].iloc[i]
        
        # Calculate number of bricks
        bricks = int(abs(price_change) / brick_size)
        
        if bricks > 0:
            direction = 1 if price_change > 0 else -1
            
            # If direction changes, add a brick in the new direction
            if direction != last_direction and last_direction != 0:
                current_brick += direction
                current_price += direction * brick_size
                renko_data.append({
                    'date': df.index[i],
                    'price': current_price,
                    'brick': current_brick,
                    'direction': direction
                })
            
            # Add bricks for the price change
            for _ in range(bricks):
                current_brick += direction
                current_price += direction * brick_size
                renko_data.append({
                    'date': df.index[i],
                    'price': current_price,
                    'brick': current_brick,
                    'direction': direction
                })
            
            last_direction = direction
    
    # Create DataFrame and ensure it's not empty
    renko_df = pd.DataFrame(renko_data)
    if len(renko_df) == 0:
        # If no bricks were created, add at least one brick
        renko_df = pd.DataFrame([{
            'date': df.index[0],
            'price': df['Close'].iloc[0],
            'brick': 0,
            'direction': 1 if df['Close'].iloc[-1] > df['Close'].iloc[0] else -1
        }])
    
    return renko_df

def plot_renko_chart(renko_df):
    """Create Renko chart using Plotly"""
    fig = go.Figure()
    
    # Plot up bricks
    up_bricks = renko_df[renko_df['direction'] == 1]
    if not up_bricks.empty:
        fig.add_trace(go.Scatter(
            x=up_bricks['date'],
            y=up_bricks['price'],
            mode='lines+markers',
            name='Up',
            line=dict(color='green', width=2),
            marker=dict(color='green', size=8)
        ))
    
    # Plot down bricks
    down_bricks = renko_df[renko_df['direction'] == -1]
    if not down_bricks.empty:
        fig.add_trace(go.Scatter(
            x=down_bricks['date'],
            y=down_bricks['price'],
            mode='lines+markers',
            name='Down',
            line=dict(color='red', width=2),
            marker=dict(color='red', size=8)
        ))
    
    fig.update_layout(
        title='Renko Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    st.title("Stock Price Prediction using Transformer")
    
    # Initialize training history at the start
    training_history = load_training_history()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        selection = st.selectbox(
            "Select Stock",
            options=get_stock_options(),
            index=0
        )
        stock_code = get_stock_code(selection)
        
        # Timeframe filter
        timeframe_options = {
            "1 Week": timedelta(days=7),
            "1 Month": timedelta(days=30),
            "3 Months": timedelta(days=90),
            "6 Months": timedelta(days=180),
            "1 Year": timedelta(days=365),
            "2 Years": timedelta(days=730),
            "5 Years": timedelta(days=1825),
            "Max": None 
        }
        selected_timeframe = st.selectbox("Select Timeframe", list(timeframe_options.keys()), index=6) # Default to 1 Year
        
        end_date = datetime.now()
        if selected_timeframe == "Max":
            start_date = datetime(2010, 1, 1)  # Default start date for max range
        else:
            start_date = end_date - timeframe_options[selected_timeframe]
        date_range = (start_date, end_date)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Analysis", "Technical Indicators", "Renko Chart", 
        "Predictions", "Training History"
    ])
    
    # Load data with timeout
    try:
        with st.spinner("Fetching stock data..."):
            ticker = yf.Ticker(f"{stock_code}.NS")
            
            # Add timeout for data fetching using threading
            import threading
            import queue
            
            def fetch_data_with_timeout():
                try:
                    df = ticker.history(start=date_range[0], end=date_range[1])
                    return df
                except Exception as e:
                    return e
            
            # Create a queue to store the result
            result_queue = queue.Queue()
            
            # Create and start the thread
            thread = threading.Thread(target=lambda: result_queue.put(fetch_data_with_timeout()))
            thread.start()
            
            # Wait for the thread to complete or timeout
            thread.join(timeout=30)
            
            if thread.is_alive():
                st.error("Data fetching timed out. Please check your internet connection and try again.")
                return
            
            # Get the result from the queue
            result = result_queue.get()
            if isinstance(result, Exception):
                st.error(f"Error fetching data: {str(result)}")
                return
            
            df = result
            
            if len(df) == 0:
                st.error("No data available for the selected date range. Try a different timeframe or stock.")
                return
            
            # Log data shape for debugging
            st.write(f"DEBUG: Data shape: {df.shape}")
            
            # Fetch stock info with timeout
            def fetch_stock_info_with_timeout():
                try:
                    return ticker.info
                except Exception as e:
                    return e
            
            result_queue = queue.Queue()
            thread = threading.Thread(target=lambda: result_queue.put(fetch_stock_info_with_timeout()))
            thread.start()
            thread.join(timeout=30)
            
            if thread.is_alive():
                st.warning("Stock info fetching timed out. Continuing with limited information.")
                stock_info = {}
            else:
                result = result_queue.get()
                if isinstance(result, Exception):
                    st.warning(f"Error fetching stock info: {str(result)}")
                    stock_info = {}
                else:
                    stock_info = result
            
            # Prepare data
            df, price_scaler, indicator_scaler = prepare_data(df)
            
            # Log prepared data shape
            st.write(f"DEBUG: Prepared data shape: {df.shape}")
            
            # Centralized check for data availability and scaler presence
            if df.empty or price_scaler is None or indicator_scaler is None:
                st.error("Not enough valid data for the selected timeframe after preprocessing. Please choose a longer timeframe or a different stock.")
                return
            
            # Ensure enough data for sequence length after preparation
            if len(df) < seq_length + 1: # +1 because we need target for prediction
                st.error(f"Not enough data for the selected timeframe after preparing indicators. Requires at least {seq_length + 1} data points. Please select a longer timeframe or a different stock.")
                return

            with tab1:
                st.subheader("Price Analysis")
                
                # Display stock metadata
                st.write("**Company Overview**")
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                with col_info2:
                    st.write(f"**Market Cap:** ₹{stock_info.get('marketCap', 0):,.0f}")
                    st.write(f"**P/E Ratio:** {stock_info.get('trailingPE', 'N/A'):.2f}")
                with col_info3:
                    st.write(f"**Beta:** {stock_info.get('beta', 'N/A'):.2f}")
                    st.write(f"**Dividend Yield:** {stock_info.get('dividendYield', 0)*100:.2f}%")
                st.markdown("--- (Data from Yahoo Finance) ---")

                fig = plot_stock_data(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics
                # Only display if df has enough data for these calculations
                if len(df) >= 2: # At least two points for daily return
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                    with col2:
                        daily_return = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                        st.metric("Daily Return", f"{daily_return:.2f}%")
                    with col3:
                        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                    with col4:
                        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                else:
                    st.info("Not enough data for key price metrics in this timeframe.")
            
            with tab2:
                st.subheader("Technical Indicators")
                # Only display if df has enough data for these calculations (already handled by prepare_data's dropna)
                if not df.empty:
                    # MACD
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'))
                    fig_macd.update_layout(title='MACD', height=400)
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Bollinger Bands
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='Upper Band', line=dict(dash='dash')))
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='Lower Band', line=dict(dash='dash')))
                    fig_bb.update_layout(title='Bollinger Bands', height=400)
                    st.plotly_chart(fig_bb, use_container_width=True)
                else:
                    st.info("Not enough data to display technical indicators in this timeframe.")
            
            with tab3:
                st.subheader("Renko Chart Analysis")
                if not df.empty:
                    # Move brick size slider to Renko Chart tab
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        brick_size = st.slider("Brick Size", min_value=1, max_value=10, value=2)
                    
                    # Calculate Renko data
                    renko_df = calculate_renko(df, brick_size)
                    
                    if not renko_df.empty:
                        # Plot Renko chart
                        fig = plot_renko_chart(renko_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display Renko statistics
                        st.subheader("Renko Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Up Bricks", len(renko_df[renko_df['direction'] == 1]))
                        with col2:
                            st.metric("Down Bricks", len(renko_df[renko_df['direction'] == -1]))
                        with col3:
                            st.metric("Net Bricks", len(renko_df[renko_df['direction'] == 1]) - len(renko_df[renko_df['direction'] == -1]))
                        
                        # Show recent moves
                        st.subheader("Recent Moves")
                        recent_moves = renko_df.tail(10).copy()
                        recent_moves['direction'] = recent_moves['direction'].map({1: '🔼', -1: '🔽'})
                        st.dataframe(recent_moves[['date', 'direction']].style.format({
                            'date': lambda x: x.strftime('%Y-%m-%d'),
                            'direction': lambda x: '🔼' if x == 1 else '🔽'
                        }))
                    else:
                        st.info("Not enough data to generate Renko chart for this timeframe/brick size.")
                else:
                    st.info("Not enough data to display Renko chart in this timeframe.")
            
            with tab4:
                st.subheader("Transformer Model Predictions")
                with st.spinner("Generating predictions..."):
                    try:
                        # Load model with timeout
                        def load_model_with_timeout():
                            try:
                                return load_model()
                            except Exception as e:
                                return e
                        
                        result_queue = queue.Queue()
                        thread = threading.Thread(target=lambda: result_queue.put(load_model_with_timeout()))
                        thread.start()
                        thread.join(timeout=30)
                        
                        if thread.is_alive():
                            st.error("Model loading timed out. Please try again.")
                            return
                        
                        result = result_queue.get()
                        if isinstance(result, Exception):
                            st.error(f"Error loading model: {str(result)}")
                            return
                        
                        model, device = result
                        
                        # Prepare sequences
                        X_sequence_data = df[features].values
                        st.write(f"DEBUG: X_sequence_data shape: {X_sequence_data.shape}")
                        
                        X = []
                        dates_for_actual = []
                        actual_prices_for_plot = []
                        
                        for i in range(len(X_sequence_data) - seq_length):
                            X.append(X_sequence_data[i:i+seq_length])
                            dates_for_actual.append(df.index[i+seq_length])
                            actual_prices_for_plot.append(df['Close'].iloc[i+seq_length])
                        
                        X = np.array(X)
                        st.write(f"DEBUG: X shape: {X.shape}")
                        X = torch.FloatTensor(X).to(device)
                        
                        if len(X) == 0:
                            st.warning(f"No full sequences can be formed for predictions with seq_length={seq_length}. Try a longer timeframe.")
                            return
                        
                        model.to(device)
                        model.train() # Set model to training mode to enable dropout during inference for MCD
                        
                        num_monte_carlo_runs = 50
                        all_predictions = []
                        
                        # Add progress bar for MCD runs
                        progress_bar = st.progress(0)
                        for run_idx in range(num_monte_carlo_runs):
                            temp_predictions = []
                            with torch.no_grad():
                                for i in range(len(X)):
                                    pred = model(X[i:i+1])
                                    temp_predictions.append(pred[:, -1, 0].item())
                            all_predictions.append(temp_predictions)
                            progress_bar.progress((run_idx + 1) / num_monte_carlo_runs)
                        
                        model.eval()
                        
                        all_predictions = np.array(all_predictions)
                        st.write(f"DEBUG: all_predictions shape: {all_predictions.shape}")
                        
                        mean_predictions = np.mean(all_predictions, axis=0)
                        lower_bound = np.percentile(all_predictions, 2.5, axis=0)
                        upper_bound = np.percentile(all_predictions, 97.5, axis=0)
                        
                        # Inverse transform predictions and bounds
                        # Ensure price_scaler is not None before using it
                        if price_scaler is not None:
                            mean_predictions = price_scaler.inverse_transform(np.c_[np.zeros((len(mean_predictions), 3)), mean_predictions.reshape(-1,1), np.zeros((len(mean_predictions), 1))])[:, 3]
                            lower_bound = price_scaler.inverse_transform(np.c_[np.zeros((len(lower_bound), 3)), lower_bound.reshape(-1,1), np.zeros((len(lower_bound), 1))])[:, 3]
                            upper_bound = price_scaler.inverse_transform(np.c_[np.zeros((len(upper_bound), 3)), upper_bound.reshape(-1,1), np.zeros((len(upper_bound), 1))])[:, 3]
                        else:
                            st.error("Price scaler not available. Cannot inverse transform predictions.")
                            return
                        
                        # Ensure actual_prices_for_plot matches the length of predictions
                        actual_prices_for_plot = np.array(actual_prices_for_plot[-len(mean_predictions):])
                        dates_for_actual = dates_for_actual[-len(mean_predictions):]

                        # Define horizons for multi-horizon predictions (re-calculate based on current X length)
                        future_horizons = {
                            "Next Day": 1,
                            "Next 5 Days (1 Week)": 5,
                            "Next 20 Days (1 Month)": 20,
                        }
                        max_future_steps = max(future_horizons.values())

                        # Prepare initial sequence for future predictions
                        current_sequence_df = df[features].iloc[-seq_length:].copy()
                        current_sequence_tensor = torch.FloatTensor(current_sequence_df.values).to(device).unsqueeze(0)

                        future_dates = []
                        future_mean_preds = []
                        future_lower_bounds = []
                        future_upper_bounds = []

                        for step in range(max_future_steps):
                            step_predictions = []
                            model.train()
                            for _ in range(num_monte_carlo_runs):
                                with torch.no_grad():
                                    pred = model(current_sequence_tensor)
                                    # Take the last prediction value
                                    step_predictions.append(pred[:, -1, 0].item())
                            model.eval()

                            step_predictions_np = np.array(step_predictions)
                            mean_pred_step = np.mean(step_predictions_np)
                            lower_bound_step = np.percentile(step_predictions_np, 2.5)
                            upper_bound_step = np.percentile(step_predictions_np, 97.5)

                            # Inverse transform the single step prediction
                            if price_scaler is not None: # Check scaler before using
                                mean_pred_step_inv = price_scaler.inverse_transform(np.array([[0,0,0,mean_pred_step,0]]))[:,3][0]
                                lower_bound_step_inv = price_scaler.inverse_transform(np.array([[0,0,0,lower_bound_step,0]]))[:,3][0]
                                upper_bound_step_inv = price_scaler.inverse_transform(np.array([[0,0,0,upper_bound_step,0]]))[:,3][0]
                            else:
                                st.error("Price scaler not available for future predictions.")
                                return

                            future_mean_preds.append(mean_pred_step_inv)
                            future_lower_bounds.append(lower_bound_step_inv)
                            future_upper_bounds.append(upper_bound_step_inv)

                            last_known_date = dates_for_actual[-1] if not future_dates else future_dates[-1]
                            next_date = last_known_date + timedelta(days=1)
                            while next_date.weekday() > 4:
                                next_date += timedelta(days=1)
                            future_dates.append(next_date)

                            next_input_row_values = current_sequence_df.iloc[-1].values.copy()
                            next_input_row_values[3] = mean_pred_step  # Close price is at index 3 in features list
                            
                            for i, feat in enumerate(features):
                                if feat != 'Close':
                                    next_input_row_values[i] = current_sequence_df.iloc[-1][feat]
                                
                            next_input_row_df = pd.DataFrame([next_input_row_values], columns=features)
                            current_sequence_df = pd.concat([current_sequence_df.iloc[1:], next_input_row_df], ignore_index=True)
                            current_sequence_tensor = torch.FloatTensor(current_sequence_df.values).to(device).unsqueeze(0)

                        combined_dates = list(dates_for_actual) + future_dates
                        combined_actual_prices = list(actual_prices_for_plot) # No actual prices for future
                        combined_mean_predictions = list(mean_predictions) + future_mean_preds
                        combined_lower_bounds = list(lower_bound) + future_lower_bounds
                        combined_upper_bounds = list(upper_bound) + future_upper_bounds

                        # --- Overlay Chart ---
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=combined_dates, y=combined_actual_prices, mode='lines', name='Actual', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=combined_dates[-len(combined_mean_predictions):], y=combined_mean_predictions, mode='lines', name='Predicted Mean', line=dict(color='orange', width=2)))

                        fig.add_trace(go.Scatter(
                            x=np.concatenate([combined_dates[-len(combined_mean_predictions):], combined_dates[-len(combined_mean_predictions):][::-1]]),
                            y=np.concatenate([combined_upper_bounds, combined_lower_bounds[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval',
                            showlegend=True
                        ))

                        fig.add_trace(go.Scatter(x=[combined_dates[-1]], y=[combined_mean_predictions[-1]], mode='markers', name='Latest Future Prediction', marker=dict(color='red', size=12, symbol='star')))

                        fig.update_layout(
                            title='Actual vs. Predicted Closing Prices with 95% Confidence Interval',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            yaxis2=dict(overlaying='y', side='right', showgrid=False, visible=False),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display multi-horizon predictions
                        st.subheader("Multi-Horizon Predictions")
                        multi_pred_cols = st.columns(len(future_horizons))
                        for i, (horizon_name, steps) in enumerate(future_horizons.items()):
                            with multi_pred_cols[i]:
                                if steps <= max_future_steps:
                                    pred_val = future_mean_preds[steps - 1]
                                    lower_val = future_lower_bounds[steps - 1]
                                    upper_val = future_upper_bounds[steps - 1]
                                    st.metric(f"{horizon_name}", f"₹{pred_val:.2f}")
                                    st.write(f"<small>Range: ₹{lower_val:.2f} - ₹{upper_val:.2f}</small>", unsafe_allow_html=True)
                                else:
                                    st.metric(f"{horizon_name}", "N/A")
                                    st.write("<small>Not enough data for prediction</small>", unsafe_allow_html=True)

                        # Calculate metrics (use mean_predictions for metrics, and actuals from plot_data)
                        mse = np.mean((mean_predictions - actual_prices_for_plot) ** 2)
                        mae = np.mean(np.abs(mean_predictions - actual_prices_for_plot))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse:.2f}")
                        with col2:
                            st.metric("Mean Absolute Error", f"{mae:.2f}")

                        # Indicator snapshot for the latest prediction (use original df and latest_idx)
                        indicator_cols = ['RSI', 'MACD', 'EMA12', 'EMA26']
                        latest_idx = len(df) - 1 # Get last index of original df
                        if latest_idx >= 0:
                            latest_indicators = {col: df[col].values[latest_idx] for col in indicator_cols}

                            st.subheader('Latest Indicator Snapshot')
                            st.markdown('<div class="compact-metrics">', unsafe_allow_html=True)
                            col1, col2, col3, col4 = st.columns(4)
                            col1.markdown(f'RSI {info_icon("RSI")}', unsafe_allow_html=True)
                            col1.metric("", f"{latest_indicators['RSI']:.2f}")
                            col2.markdown(f'MACD {info_icon("MACD")}', unsafe_allow_html=True)
                            col2.metric("", f"{latest_indicators['MACD']:.2f}")
                            col3.markdown(f'EMA12 {info_icon("EMA12")}', unsafe_allow_html=True)
                            col3.metric("", f"{latest_indicators['EMA12']:.2f}")
                            col4.markdown(f'EMA26 {info_icon("EMA26")}', unsafe_allow_html=True)
                            col4.metric("", f"{latest_indicators['EMA26']:.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.warning("Not enough data for Latest Indicator Snapshot.")

                        # Textual explanation for the latest prediction
                        if latest_idx >= 0:
                            explanation = []
                            if latest_indicators['RSI'] > 70:
                                explanation.append("RSI is high (>70), indicating the stock may be overbought and due for a pullback.")
                            elif latest_indicators['RSI'] < 30:
                                explanation.append("RSI is low (<30), indicating the stock may be oversold and could rebound.")
                            else:
                                explanation.append(f"RSI is neutral at {latest_indicators['RSI']:.1f}.")

                            if latest_indicators['MACD'] > 0:
                                explanation.append("MACD is positive, suggesting bullish momentum.")
                            else:
                                explanation.append("MACD is negative, suggesting bearish momentum.")

                            if latest_indicators['EMA12'] > latest_indicators['EMA26']:
                                explanation.append("EMA12 is above EMA26, indicating a short-term uptrend.")
                            else:
                                explanation.append("EMA12 is below EMA26, indicating a short-term downtrend.")

                            st.markdown(f"**Model's Reasoning:**<br>" + '<br>'.join(explanation), unsafe_allow_html=True)
                        else:
                            st.warning("Not enough data for Model's Reasoning.")

                        # Feature importance (permutation importance proxy)
                        if len(X_sequence_data) >= seq_length + 1: # Ensure enough data to form at least one sequence and get last features
                            base_features_for_importance = df[features].iloc[-seq_length:].copy()
                            original_last_prediction = mean_predictions[-1]

                            importances = []
                            for col in indicator_cols:
                                perturbed_sequence_df = current_sequence_df.copy()
                                perturbed_sequence_df[col] = np.random.permutation(perturbed_sequence_df[col].values)
                                perturbed_sequence_tensor = torch.FloatTensor(perturbed_sequence_df.values).to(device).unsqueeze(0)
                                
                                perturbed_predictions = []
                                model.train()
                                for _ in range(num_monte_carlo_runs):
                                    with torch.no_grad():
                                        pred = model(perturbed_sequence_tensor)
                                        perturbed_predictions.append(pred[:, -1, 0].item())
                                model.eval()

                                mean_perturbed_pred = np.mean(perturbed_predictions)
                                # Ensure price_scaler is not None before using it
                                if price_scaler is not None:
                                    mean_perturbed_pred_inv = price_scaler.inverse_transform(np.array([[0,0,0,mean_perturbed_pred,0]]))[:,3][0]
                                else:
                                    st.error("Price scaler not available for feature importance calculation.")
                                    return
                                
                                impact = abs(mean_perturbed_pred_inv - actual_prices_for_plot[-1])
                                importances.append(impact)

                            importance_df = pd.DataFrame({'Indicator': indicator_cols, 'Importance': importances})
                            importance_df = importance_df.sort_values('Importance', ascending=False)

                            st.subheader('Feature Importance (Proxy)')
                            fig_imp = px.bar(importance_df, x='Importance', y='Indicator', orientation='h',
                                             labels={'Importance': 'Impact on Prediction', 'Indicator': 'Indicator'},
                                             height=220)
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.warning("Not enough data to calculate Feature Importance.")

                        # Prepare data for the detailed prediction table
                        last_n_table = len(mean_predictions)
                        if last_n_table > 0:
                            table_df = pd.DataFrame({
                                'Date': dates_for_actual[-last_n_table:],
                                'Actual': actual_prices_for_plot[-last_n_table:],
                                'Predicted Mean': mean_predictions[-last_n_table:],
                                'Lower Bound': lower_bound[-last_n_table:],
                                'Upper Bound': upper_bound[-last_n_table:],
                            })
                            table_df['Error'] = table_df['Actual'] - table_df['Predicted Mean']
                            for col in indicator_cols:
                                table_df[col] = df[col].values[-last_n_table:]

                            st.subheader('Prediction Details')
                            st.dataframe(
                                table_df.style.format({
                                    'Actual': '₹{:.2f}',
                                    'Predicted Mean': '₹{:.2f}',
                                    'Lower Bound': '₹{:.2f}',
                                    'Upper Bound': '₹{:.2f}',
                                    'Error': '₹{:.2f}',
                                    'RSI': '{:.2f}',
                                    'MACD': '{:.2f}',
                                    'EMA12': '{:.2f}',
                                    'EMA26': '{:.2f}'
                                })
                            )

                            csv = table_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Prediction Table as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("Not enough data for Prediction Details table.")

                        # Update training history
                        if stock_code not in training_history:
                            training_history[stock_code] = []
                        
                        training_history[stock_code].append({
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'train_loss': float(mse),  # Using MSE as train loss
                            'val_loss': float(mae),    # Using MAE as val loss
                            'accuracy': float(100 * (1 - mae/df['Close'].mean())),  # Simple accuracy metric
                            'data_points': len(df)
                        })
                        
                        save_training_history(training_history)
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.write("DEBUG: Full error traceback:")
                        import traceback
                        st.code(traceback.format_exc())
                        return
        
        with tab5:
            st.subheader("Training History")
            if stock_code in training_history and training_history[stock_code]:
                history = training_history[stock_code]
                fig = plot_training_metrics(history)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display training history table
                history_df = pd.DataFrame(history)
                history_df['date'] = pd.to_datetime(history_df['date'])
                history_df = history_df.sort_values('date', ascending=False)
                
                st.dataframe(
                    history_df.style.format({
                        'train_loss': '{:.4f}',
                        'val_loss': '{:.4f}',
                        'accuracy': '{:.2f}%',
                        'data_points': '{:,.0f}'
                    })
                )
            else:
                st.info("No training history available for this stock.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 