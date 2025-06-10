import unittest
import torch
import numpy as np
from src.models.transformer import StockTransformer

class TestStockTransformer(unittest.TestCase):
    def setUp(self):
        self.input_size = 15
        self.batch_size = 32
        self.seq_length = 30
        self.model = StockTransformer(input_size=self.input_size)
        
    def test_model_initialization(self):
        self.assertIsInstance(self.model, StockTransformer)
        self.assertEqual(self.model.d_model, 32)  # Default value
        
    def test_forward_pass(self):
        # Create sample input
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, 1))
        
    def test_from_config(self):
        config = {
            'input_size': self.input_size,
            'd_model': 64,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.2
        }
        
        model = StockTransformer.from_config(config)
        
        # Check if model was created with correct parameters
        self.assertEqual(model.d_model, config['d_model'])
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, 1))
        
    def test_positional_encoding(self):
        # Create sample input
        x = torch.randn(self.batch_size, self.seq_length, self.model.d_model)
        
        # Get positional encoding
        pe = self.model.pos_encoder(x)
        
        # Check if positional encoding was added
        self.assertEqual(pe.shape, x.shape)
        self.assertFalse(torch.allclose(pe, x))  # Should be different due to positional encoding
        
if __name__ == '__main__':
    unittest.main() 