import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Setting pe shape explicitly to (max_len, 1, d_model) as per saved model
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        logger.debug(f"PositionalEncoding initialized with pe shape: {self.pe.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (max_len, 1, d_model)
        return x + self.pe[:x.size(1), :]

class StockTransformer(nn.Module):
    def __init__(self, 
                 input_size: int,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection (RESTORED to match the OLDER SAVED MODEL ARCHITECTURE)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        # dim_feedforward MUST BE 128 (d_model * 4) to match the saved model
        _dim_feedforward = d_model * 4 
        logger.debug(f"StockTransformer initializing with dim_feedforward: {_dim_feedforward}")
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=_dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output projection (RESTORED to match the OLDER SAVED MODEL ARCHITECTURE)
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.output_projection(x)  # (batch_size, seq_len, 1)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StockTransformer':
        """Create model from configuration dictionary"""
        return cls(
            input_size=config['input_size'],
            d_model=config.get('d_model', 32),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            max_len=config.get('max_len', 5000)
        ) 