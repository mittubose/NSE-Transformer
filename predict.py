import torch
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from transformer_model import StockTransformer
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['MA20'] + (df['STD20'] * 2)
    df['LowerBand'] = df['MA20'] - (df['STD20'] * 2)
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    return df

def prepare_data(df):
    """Prepare data for prediction"""
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    indicator_features = ['RSI', 'MACD', 'Signal', 'MA5', 'MA10', 'MA20', 'UpperBand', 'LowerBand']
    
    # Normalize data
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    
    price_scaler = RobustScaler()
    indicator_scaler = MinMaxScaler()
    
    df[price_features] = price_scaler.fit_transform(df[price_features])
    df[indicator_features] = indicator_scaler.fit_transform(df[indicator_features])
    
    return df, price_scaler, indicator_scaler

def plot_predictions(actual, predicted, dates):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # Model parameters
    symbol = 'RELIANCE'
    seq_length = 30
    d_model = 32
    nhead = 4
    num_layers = 2
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    model = StockTransformer(
        input_dim=15,  # Number of features
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        logger.info('Successfully loaded model')
    except Exception as e:
        logger.error(f'Error loading model: {str(e)}')
        return
    
    model.to(device)
    model.eval()
    
    # Fetch recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)  # Get 100 days of data
    
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start_date, end=end_date)
        logger.info(f'Successfully fetched data for {symbol}')
    except Exception as e:
        logger.error(f'Error fetching data: {str(e)}')
        return
    
    # Prepare data
    df, price_scaler, indicator_scaler = prepare_data(df)
    
    # Create sequences
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'RSI', 'MACD', 'Signal', 'MA5', 'MA10', 
                'MA20', 'UpperBand', 'LowerBand', 'EMA12', 'EMA26']
    
    X = []
    dates = []
    actual_prices = []
    
    for i in range(len(df) - seq_length):
        X.append(df[features].iloc[i:i+seq_length].values)
        dates.append(df.index[i+seq_length])
        actual_prices.append(df['Close'].iloc[i+seq_length])
    
    # Convert to numpy array first, then to tensor
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
    # Create a dummy array with the same shape as the original data
    dummy_array = np.zeros((len(predictions), 5))  # 5 for price features
    dummy_array[:, 3] = predictions[:, 0]  # Put predictions in the Close price column
    predictions = price_scaler.inverse_transform(dummy_array)[:, 3]  # Get only the Close price column
    
    # Inverse transform actual prices
    actual_prices = np.array(actual_prices).reshape(-1, 1)
    dummy_array = np.zeros((len(actual_prices), 5))
    dummy_array[:, 3] = actual_prices[:, 0]
    actual_prices = price_scaler.inverse_transform(dummy_array)[:, 3]
    
    # Calculate metrics
    mse = np.mean((predictions - actual_prices) ** 2)
    mae = np.mean(np.abs(predictions - actual_prices))
    
    logger.info(f'Mean Squared Error: {mse:.2f}')
    logger.info(f'Mean Absolute Error: {mae:.2f}')
    
    # Plot results
    plot_predictions(actual_prices, predictions, dates)
    logger.info('Predictions plot saved as predictions.png')
    
    # Print last 5 predictions
    logger.info('\nLast 5 predictions:')
    for i in range(-5, 0):
        logger.info(f'Date: {dates[i].strftime("%Y-%m-%d")}')
        logger.info(f'Actual: {actual_prices[i]:.2f}')
        logger.info(f'Predicted: {predictions[i]:.2f}')
        logger.info('---')

if __name__ == '__main__':
    main() 