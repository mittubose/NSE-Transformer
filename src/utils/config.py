from typing import Dict, Any
import yaml
import os

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            return self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        config = {
            'model': {
                'input_size': 15,  # Number of features
                'd_model': 32,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'max_len': 5000
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.0001,
                'early_stopping_patience': 5
            },
            'data': {
                'seq_length': 30,
                'train_split': 0.8
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return config
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config['data']
        
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        self.config = deep_update(self.config, updates)
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False) 