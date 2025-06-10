import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from transformer_model import StockTransformer
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import argparse
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        # Add .NS suffix for NSE stocks
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(start=start_date, end=end_date)
        logger.info(f"Successfully fetched data for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

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

def prepare_sequences(data, seq_length):
    """Prepare sequences for transformer input"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data.iloc[i + seq_length]['Close']
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def plot_losses(train_losses, val_losses, save_path='training_losses.png'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def log_training_session(symbol, start_date, end_date, train_loss, val_loss, model_path, history_path='training_history.json'):
    record = {
        'timestamp': str(date.today()),
        'symbol': symbol,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_path': model_path
    }
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    else:
        history = []
    history.append(record)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

def prepare_data(df):
    """Prepare and clean the data"""
    # Calculate returns and log returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f'Dropped {initial_rows - len(df)} rows with NaN values')
    
    # Normalize the data
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'Returns', 'Log_Returns',
                      'RSI', 'BB_high', 'BB_low', 'BB_mid',
                      'MACD', 'MACD_signal', 'SMA_20', 'SMA_50']
    
    # Use RobustScaler for price data and MinMaxScaler for indicators
    price_scaler = RobustScaler()
    indicator_scaler = MinMaxScaler()
    
    # Scale price-related features
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[price_features] = price_scaler.fit_transform(df[price_features])
    
    # Scale indicators
    indicator_features = [col for col in feature_columns if col not in price_features]
    df[indicator_features] = indicator_scaler.fit_transform(df[indicator_features])
    
    return df, feature_columns, price_scaler, indicator_scaler

def main(symbol='RELIANCE', days=730, model_path='best_model.pth'):
    seq_length = 30
    batch_size = 32
    d_model = 32
    nhead = 4
    num_layers = 2
    num_epochs = 50
    learning_rate = 0.0001

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        df = fetch_stock_data(symbol, start_date, end_date)
        logger.info(f'Initial data shape: {df.shape}')
        df, feature_columns, price_scaler, indicator_scaler = prepare_data(df)
        logger.info(f'Final data shape: {df.shape}')
        X, y = prepare_sequences(df[feature_columns], seq_length)
        logger.info(f'X shape: {X.shape}, y shape: {y.shape}')
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StockTransformer(
            input_size=len(feature_columns),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                output = output.squeeze()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    output = output.squeeze()
                    val_loss += criterion(output, target).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            logger.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
                logger.info(f'New best model saved with validation loss: {best_val_loss:.6f}')
        plot_losses(train_losses, val_losses)
        log_training_session(symbol, start_date, end_date, float(train_losses[-1]), float(val_losses[-1]), model_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Transformer model for stock prediction.')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol (without .NS)')
    parser.add_argument('--days', type=int, default=730, help='Number of days of historical data to use')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save the trained model')
    args = parser.parse_args()
    main(symbol=args.symbol, days=args.days, model_path=args.model_path) 