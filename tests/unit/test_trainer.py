import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from src.models.transformer import StockTransformer
from src.training.trainer import Trainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.input_size = 15
        self.batch_size = 32
        self.seq_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = StockTransformer(input_size=self.input_size).to(self.device)
        
        # Create criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # Create sample data
        self.X = torch.randn(100, self.seq_length, self.input_size)
        self.y = torch.randn(100, 1)
        
        # Create data loaders
        self.train_dataset = torch.utils.data.TensorDataset(self.X, self.y)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
    def test_train_epoch(self):
        # Train for one epoch
        loss = self.trainer.train_epoch(self.train_loader)
        
        # Check if loss is a float
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
        
    def test_validate(self):
        # Create validation loader
        val_dataset = torch.utils.data.TensorDataset(self.X, self.y)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Validate
        val_loss = self.trainer.validate(val_loader)
        
        # Check if loss is a float
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0)
        
    def test_train(self):
        # Create validation loader
        val_dataset = torch.utils.data.TensorDataset(self.X, self.y)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Train for a few epochs
        history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=val_loader,
            num_epochs=3
        )
        
        # Check history
        self.assertIn('train_losses', history)
        self.assertIn('val_losses', history)
        self.assertEqual(len(history['train_losses']), 3)
        self.assertEqual(len(history['val_losses']), 3)
        
    def test_log_training_session(self):
        # Log a training session
        self.trainer.log_training_session(
            symbol='RELIANCE',
            start_date=datetime.now(),
            end_date=datetime.now(),
            train_loss=0.1,
            val_loss=0.2
        )
        
        # Check if history file was created
        import os
        self.assertTrue(os.path.exists(self.trainer.history_path))
        
if __name__ == '__main__':
    unittest.main() 