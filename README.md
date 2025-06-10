# Indian Stock Market Transformer

A transformer-based model for predicting Indian stock prices using historical data and technical indicators.

## Features

- Fetches historical data from NSE (National Stock Exchange)
- Implements a transformer model with attention mechanism
- Includes technical indicators (RSI, Bollinger Bands, MACD, Moving Averages)
- Provides both training and prediction capabilities
- Comprehensive logging for monitoring and debugging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Make predictions:
```bash
python predict.py
```

## Model Architecture

- Input: Sequence of historical price data and technical indicators
- Transformer layers with multi-head attention
- Positional encoding for sequence information
- Output: Next day's predicted price

## Technical Indicators Used

- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Simple Moving Averages (20 and 50 days)

## Parameters

You can modify the following parameters in `train.py` and `predict.py`:

- `symbol`: Stock symbol (default: 'RELIANCE')
- `seq_length`: Sequence length for prediction (default: 30)
- `d_model`: Transformer model dimension (default: 64)
- `nhead`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 4)
- `batch_size`: Training batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 50)

## Notes

- The model uses 2 years of historical data for training
- Data is split into 80% training and 20% validation
- Best model is saved as 'best_model.pth'
- All predictions are logged with current price and percentage change 