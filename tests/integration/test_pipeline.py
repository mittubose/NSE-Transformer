import unittest
import torch
from datetime import datetime, timedelta
from src.data.data_loader import StockDataLoader
from src.models.transformer import StockTransformer
from src.training.trainer import Trainer
import os

class TestFullPipeline(unittest.TestCase):
    def test_end_to_end_training(self):
        symbol = 'RELIANCE'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        seq_length = 10
        batch_size = 8
        num_epochs = 2
        
        # Data loading
        loader = StockDataLoader(symbol, start_date, end_date)
        try:
            X, y, feature_columns, price_scaler, indicator_scaler = loader.get_data(seq_length)
        except ValueError as e:
            self.skipTest(f"Integration test skipped: {e}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            self.skipTest("Not enough data available for integration test. Try a longer date range or different symbol.")
        
        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StockTransformer(input_size=len(feature_columns)).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Trainer
        trainer = Trainer(model, criterion, optimizer, device)
        history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
        
        # Check training history
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertEqual(len(history['train_losses']), num_epochs)
        
        # Prediction
        model.eval()
        with torch.no_grad():
            sample = torch.FloatTensor(X[:1]).to(device)
            pred = model(sample)
            self.assertEqual(pred.shape, (1, seq_length, 1))
        
        # Clean up
        if os.path.exists(trainer.model_save_path):
            os.remove(trainer.model_save_path)

if __name__ == '__main__':
    unittest.main() 