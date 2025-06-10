import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        
        # Input embedding with layer normalization
        self.embedding = nn.Linear(input_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder with layer normalization
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Reduced from 4
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        # Input embedding and normalization
        src = self.embedding(src)
        src = self.norm1(src)
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src, src_mask)
        
        # Take only the last time step
        output = output[:, -1, :]
        
        # Output layers with residual connection
        residual = output
        output = self.fc1(output)
        output = self.norm2(output)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output 