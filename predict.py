import torch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from transformer_model import StockTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, input_dim, d_model, nhead, num_layers):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def prepare_prediction_data(df, feature_columns, scaler, seq_length):
    """Prepare data for prediction"""
    scaled_data = scaler.transform(df[feature_columns])
    sequence = scaled_data[-seq_length:]
    return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

def predict_next_day(model, data, scaler, feature_columns):
    """Make prediction for the next day"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    with torch.no_grad():
        prediction = model(data)
        prediction = prediction.cpu().numpy()
        
        # Inverse transform the prediction
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, feature_columns.index('Close')] = prediction[0, -1, 0]
        unscaled_prediction = scaler.inverse_transform(dummy_array)[0, feature_columns.index('Close')]
        
        return unscaled_prediction

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
    
    return df

def main():
    # Parameters
    symbol = 'RELIANCE'
    seq_length = 30
    d_model = 64
    nhead = 8
    num_layers = 4
    model_path = 'best_model.pth'
    
    try:
        # Fetch recent data
        end_date = date.today()
        start_date = end_date - timedelta(days=seq_length + 10)  # Extra days for technical indicators
        
        # Add .NS suffix for NSE stocks
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start_date, end=end_date)
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Prepare features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'RSI', 'BB_high', 'BB_low', 'BB_mid',
                          'MACD', 'MACD_signal', 'SMA_20', 'SMA_50']
        
        # Scale the data
        scaler = MinMaxScaler()
        scaler.fit(df[feature_columns])
        
        # Prepare prediction data
        prediction_data = prepare_prediction_data(df, feature_columns, scaler, seq_length)
        
        # Load model and make prediction
        model = load_model(model_path, len(feature_columns), d_model, nhead, num_layers)
        next_day_prediction = predict_next_day(model, prediction_data, scaler, feature_columns)
        
        logger.info(f"Current price: {df['Close'].iloc[-1]:.2f}")
        logger.info(f"Predicted price for next day: {next_day_prediction:.2f}")
        logger.info(f"Predicted change: {((next_day_prediction - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100):.2f}%")
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main() 