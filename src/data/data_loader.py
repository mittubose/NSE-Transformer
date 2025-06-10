import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.price_scaler = RobustScaler()
        self.indicator_scaler = MinMaxScaler()
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(f"{self.symbol}.NS")
            data = ticker.history(start=self.start_date, end=self.end_date)
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            logger.info(f"Successfully fetched data for {self.symbol}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {str(e)}")
            raise

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list, RobustScaler, MinMaxScaler]:
        """Prepare and clean the data"""
        # Calculate returns and log returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Drop NaN values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f'Dropped {initial_rows - len(df)} rows with NaN values')
        if df.empty:
            raise ValueError("No data left after dropping NaNs. Not enough data for scaling and training.")
        
        # Define feature columns
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'Returns', 'Log_Returns',
                          'RSI', 'BB_high', 'BB_low', 'BB_mid',
                          'MACD', 'MACD_signal', 'SMA_20', 'SMA_50']
        
        # Scale price-related features
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.loc[:, price_features] = self.price_scaler.fit_transform(df[price_features])
        
        # Scale indicators
        indicator_features = [col for col in feature_columns if col not in price_features]
        df.loc[:, indicator_features] = self.indicator_scaler.fit_transform(df[indicator_features])
        
        return df, feature_columns, self.price_scaler, self.indicator_scaler

    def prepare_sequences(self, data: pd.DataFrame, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for transformer input"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = data.iloc[i + seq_length]['Close']
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def get_data(self, seq_length: int) -> Tuple[np.ndarray, np.ndarray, list, RobustScaler, MinMaxScaler]:
        """Main method to get processed data"""
        df = self.fetch_data()
        df, feature_columns, price_scaler, indicator_scaler = self.prepare_data(df)
        X, y = self.prepare_sequences(df[feature_columns], seq_length)
        return X, y, feature_columns, price_scaler, indicator_scaler 