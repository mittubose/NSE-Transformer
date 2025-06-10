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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the transformer model"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze()  # Remove extra dimension
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = output.squeeze()  # Remove extra dimension
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                   f'Train Loss: {avg_train_loss:.6f}, '
                   f'Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'New best model saved with validation loss: {best_val_loss:.6f}')
    
    # Plot losses
    plot_losses(train_losses, val_losses)

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

def main():
    # Parameters
    symbol = 'RELIANCE'
    seq_length = 30
    batch_size = 32
    d_model = 32  # Reduced from 64
    nhead = 4     # Reduced from 8
    num_layers = 2 # Reduced from 4
    num_epochs = 50
    learning_rate = 0.0001  # Reduced from 0.001
    
    try:
        # Fetch data
        end_date = date.today()
        start_date = end_date - timedelta(days=365*2)
        
        df = fetch_stock_data(symbol, start_date, end_date)
        logger.info(f'Initial data shape: {df.shape}')
        
        # Prepare data
        df, feature_columns, price_scaler, indicator_scaler = prepare_data(df)
        logger.info(f'Final data shape: {df.shape}')
        
        # Create sequences
        X, y = prepare_sequences(df[feature_columns], seq_length)
        logger.info(f'X shape: {X.shape}, y shape: {y.shape}')
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {device}')
        
        model = StockTransformer(
            input_dim=len(feature_columns),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 