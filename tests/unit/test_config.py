import unittest
import os
import yaml
from src.utils.config import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_config_path = 'test_config.yaml'
        self.config = Config(self.test_config_path)
        
    def tearDown(self):
        # Clean up test config file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
            
    def test_default_config_creation(self):
        # Check if default config was created
        self.assertTrue(os.path.exists(self.test_config_path))
        
        # Load config file
        with open(self.test_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Check default values
        self.assertIn('model', config_data)
        self.assertIn('training', config_data)
        self.assertIn('data', config_data)
        
    def test_get_model_config(self):
        model_config = self.config.get_model_config()
        
        # Check model config structure
        self.assertIn('input_size', model_config)
        self.assertIn('d_model', model_config)
        self.assertIn('nhead', model_config)
        self.assertIn('num_layers', model_config)
        
    def test_get_training_config(self):
        training_config = self.config.get_training_config()
        
        # Check training config structure
        self.assertIn('batch_size', training_config)
        self.assertIn('num_epochs', training_config)
        self.assertIn('learning_rate', training_config)
        
    def test_get_data_config(self):
        data_config = self.config.get_data_config()
        
        # Check data config structure
        self.assertIn('seq_length', data_config)
        self.assertIn('train_split', data_config)
        
    def test_update_config(self):
        # Update config
        updates = {
            'model': {
                'd_model': 64,
                'nhead': 8
            },
            'training': {
                'learning_rate': 0.0005
            }
        }
        self.config.update_config(updates)
        
        # Check if updates were applied
        model_config = self.config.get_model_config()
        training_config = self.config.get_training_config()
        
        self.assertEqual(model_config['d_model'], 64)
        self.assertEqual(model_config['nhead'], 8)
        self.assertEqual(training_config['learning_rate'], 0.0005)
        
if __name__ == '__main__':
    unittest.main() 