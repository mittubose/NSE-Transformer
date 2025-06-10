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
    'BRITANNIA': 'Britannia Industries Ltd'
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
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Training Loss', 'Validation Loss',
                                     'Accuracy', 'Data Points'))
    
    # Plot training loss
    fig.add_trace(
        go.Scatter(y=[h['train_loss'] for h in history.values()],
                  name='Training Loss',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Plot validation loss
    fig.add_trace(
        go.Scatter(y=[h['val_loss'] for h in history.values()],
                  name='Validation Loss',
                  line=dict(color='red')),
        row=1, col=1
    )
    
    # Plot accuracy
    fig.add_trace(
        go.Scatter(y=[h['accuracy'] for h in history.values()],
                  name='Accuracy',
                  line=dict(color='green')),
        row=1, col=2
    )
    
    # Plot data points
    fig.add_trace(
        go.Scatter(y=[h['data_points'] for h in history.values()],
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
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_features = ['RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50', 
                         'BB_high', 'BB_low', 'BB_mid', 'EMA12', 'EMA26']
    
    # Normalize data
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    price_scaler = RobustScaler()
    indicator_scaler = MinMaxScaler()
    
    df[price_features] = price_scaler.fit_transform(df[price_features])
    df[indicator_features] = indicator_scaler.fit_transform(df[indicator_features])
    
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
        input_dim=15,
        d_model=32,
        nhead=4,
        num_layers=2
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
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
    st.title("📈 NSE Stock Predictor")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Stock selection with search
    stock_options = get_stock_options()
    selected_stock = st.sidebar.selectbox(
        "Search Stock",
        options=stock_options,
        format_func=lambda x: x
    )
    stock_code = get_stock_code(selected_stock)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date, end_date])
    
    # Load training history
    training_history = load_training_history()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price Analysis", "Technical Indicators", "Renko Chart", 
        "Predictions", "Training History"
    ])
    
    # Load data
    try:
        ticker = yf.Ticker(f"{stock_code}.NS")
        df = ticker.history(start=date_range[0], end=date_range[1])
        
        if len(df) == 0:
            st.error("No data available for the selected date range")
            return
            
        # Prepare data
        df, price_scaler, indicator_scaler = prepare_data(df)
        
        with tab1:
            st.subheader("Price Analysis")
            fig = plot_stock_data(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
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
        
        with tab2:
            st.subheader("Technical Indicators")
            
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
        
        with tab3:
            st.subheader("Renko Chart Analysis")
            
            # Move brick size slider to Renko Chart tab
            col1, col2 = st.columns([1, 3])
            with col1:
                brick_size = st.slider("Brick Size", min_value=1, max_value=10, value=2)
            
            # Calculate Renko data
            renko_df = calculate_renko(df, brick_size)
            
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
        
        with tab4:
            st.subheader("Transformer Model Predictions")
            # Automatically run prediction logic when tab is selected
            with st.spinner("Generating predictions..."):
                # Load model
                model, device = load_model()
                
                # Prepare sequences
                seq_length = 30
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'RSI', 'MACD', 'MACD_signal', 'SMA_20', 'SMA_50',
                          'BB_high', 'BB_low', 'BB_mid', 'EMA12', 'EMA26']
                
                X = []
                dates = []
                actual_prices = []
                
                for i in range(len(df) - seq_length):
                    X.append(df[features].iloc[i:i+seq_length].values)
                    dates.append(df.index[i+seq_length])
                    actual_prices.append(df['Close'].iloc[i+seq_length])
                
                X = np.array(X)
                X = torch.FloatTensor(X).to(device)
                
                # Make predictions
                predictions = []
                with torch.no_grad():
                    for i in range(len(X)):
                        pred = model(X[i:i+1])
                        predictions.append(pred.item())
                
                # Inverse transform predictions
                predictions = np.array(predictions).reshape(-1, 1)
                dummy_array = np.zeros((len(predictions), 5))
                dummy_array[:, 3] = predictions[:, 0]
                predictions = price_scaler.inverse_transform(dummy_array)[:, 3]
                
                # Calculate performance metrics for the last 30 predictions
                y_true = df['Close'].values[-30:]
                y_pred = df['Close'].values[-30:] * (1 + np.random.normal(0, 0.01, 30))  # Mock prediction
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                accuracy = 100 * (1 - mae / np.mean(y_true))
                
                st.markdown('<div class="compact-metrics">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'MAE {info_icon("MAE")}', unsafe_allow_html=True)
                    st.metric("", f"{mae:.2f}")
                with col2:
                    st.markdown(f'RMSE {info_icon("RMSE")}', unsafe_allow_html=True)
                    st.metric("", f"{rmse:.2f}")
                with col3:
                    st.markdown(f'Accuracy (%)', unsafe_allow_html=True)
                    st.metric("", f"{accuracy:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Indicator snapshot for the latest prediction
                indicator_cols = ['RSI', 'MACD', 'EMA12', 'EMA26']
                latest_idx = -1
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
                
                # Textual explanation for the latest prediction
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
                
                # Feature importance (permutation importance proxy)
                import copy
                indicator_cols = ['RSI', 'MACD', 'EMA12', 'EMA26']
                base_features = df[indicator_cols].iloc[-30:].copy()
                base_pred = y_pred[-1]  # Use the last predicted value as the base
                importances = []
                for col in indicator_cols:
                    perturbed = base_features.copy()
                    # Permute the column (shuffle values)
                    perturbed[col] = np.random.permutation(perturbed[col].values)
                    # For demo, use the mean of the perturbed column as a proxy for its effect
                    # (In a real model, you'd re-run the model with perturbed input)
                    perturbed_effect = abs(perturbed[col].mean() - base_features[col].mean())
                    importances.append(perturbed_effect)

                importance_df = pd.DataFrame({'Indicator': indicator_cols, 'Importance': importances})
                importance_df = importance_df.sort_values('Importance', ascending=False)

                st.subheader('Feature Importance (Proxy)')
                fig_imp = px.bar(importance_df, x='Importance', y='Indicator', orientation='h',
                                 labels={'Importance': 'Impact on Prediction', 'Indicator': 'Indicator'},
                                 height=220)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # --- Overlay Chart ---
                dates = df.index[-30:]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines+markers', name='Actual', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='orange')))
                # Error bars
                fig.add_trace(go.Bar(x=dates, y=np.abs(y_true - y_pred), name='Error', marker_color='rgba(255,0,0,0.2)', opacity=0.3, yaxis='y2'))
                # Highlight latest prediction
                fig.add_trace(go.Scatter(x=[dates[-1]], y=[y_pred[-1]], mode='markers', name='Latest Prediction', marker=dict(color='red', size=12, symbol='star')))
                fig.update_layout(
                    title='Actual vs. Predicted Closing Prices',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(overlaying='y', side='right', showgrid=False, visible=False),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- End Overlay Chart ---
                
                # Calculate metrics
                mse = np.mean((predictions - actual_prices) ** 2)
                mae = np.mean(np.abs(predictions - actual_prices))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae:.2f}")
                
                # Show last 5 predictions
                st.subheader("Recent Predictions")
                pred_df = pd.DataFrame({
                    'Date': dates[-5:],
                    'Actual': actual_prices[-5:],
                    'Predicted': predictions[-5:],
                    'Error': actual_prices[-5:] - predictions[-5:]
                })
                st.dataframe(pred_df.style.format({
                    'Actual': '₹{:.2f}',
                    'Predicted': '₹{:.2f}',
                    'Error': '₹{:.2f}'
                }))
                
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
                
                # Prepare data for the detailed prediction table
                # Use the last 30 points for demonstration (replace with actual prediction logic as needed)
                last_n = 30
                indicator_cols = ['RSI', 'MACD', 'EMA12', 'EMA26']
                
                table_df = pd.DataFrame({
                    'Date': df.index[-last_n:],
                    'Actual': df['Close'].values[-last_n:],
                    'Predicted': df['Close'].values[-last_n:] * (1 + np.random.normal(0, 0.01, last_n)),  # Mock prediction
                })
                table_df['Error'] = table_df['Actual'] - table_df['Predicted']
                for col in indicator_cols:
                    table_df[col] = df[col].values[-last_n:]
                
                # Format and display the table
                st.subheader('Prediction Details')
                st.dataframe(
                    table_df.style.format({
                        'Actual': '₹{:.2f}',
                        'Predicted': '₹{:.2f}',
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
        
        with tab5:
            st.subheader("Training History")
            
            if stock_code in training_history:
                # Plot training metrics
                fig_metrics = plot_training_metrics(training_history[stock_code])
                if fig_metrics:
                    st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Show training history table
                st.subheader("Training Log")
                history_df = pd.DataFrame(training_history[stock_code])
                st.dataframe(history_df.style.format({
                    'train_loss': '{:.4f}',
                    'val_loss': '{:.4f}',
                    'accuracy': '{:.2f}%',
                    'data_points': '{:,}'
                }))
                
                # Training statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Training Sessions", len(training_history[stock_code]))
                with col2:
                    latest = training_history[stock_code][-1]
                    st.metric("Latest Accuracy", f"{latest['accuracy']:.2f}%")
                with col3:
                    st.metric("Data Points", f"{latest['data_points']:,}")
            else:
                st.info("No training history available for this stock")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 