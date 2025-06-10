import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_loader import StockDataLoader

class TestStockDataLoader(unittest.TestCase):
    def setUp(self):
        self.symbol = 'RELIANCE'
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        self.loader = StockDataLoader(self.symbol, self.start_date, self.end_date)
        
    def test_fetch_data(self):
        df = self.loader.fetch_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Close', df.columns)
        
    def test_add_technical_indicators(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 100,
            'Low': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add indicators
        df_with_indicators = self.loader.add_technical_indicators(df)
        
        # Check if indicators were added
        self.assertIn('RSI', df_with_indicators.columns)
        self.assertIn('MACD', df_with_indicators.columns)
        self.assertIn('BB_high', df_with_indicators.columns)
        
    def test_prepare_data(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 100,
            'Low': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Prepare data
        df_processed, feature_columns, price_scaler, indicator_scaler = self.loader.prepare_data(df)
        
        # Check if data was processed correctly
        self.assertIsInstance(df_processed, pd.DataFrame)
        self.assertIsInstance(feature_columns, list)
        self.assertFalse(df_processed.isna().any().any())
        
    def test_prepare_sequences(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 100,
            'Low': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Prepare sequences
        seq_length = 10
        X, y = self.loader.prepare_sequences(df, seq_length)
        
        # Check shapes
        self.assertEqual(X.shape[1], seq_length)
        self.assertEqual(len(X), len(y))
        
if __name__ == '__main__':
    unittest.main() 