import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 model_save_path: str = 'best_model.pth',
                 history_path: str = 'training_history.json'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_save_path = model_save_path
        self.history_path = history_path
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            # Only use the last output for loss
            output = output[:, -1, 0] if output.dim() == 3 else output[:, -1]
            target = target.squeeze() if target.dim() > 1 else target
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # Only use the last output for loss
                output = output[:, -1, 0] if output.dim() == 3 else output[:, -1]
                target = target.squeeze() if target.dim() > 1 else target
                val_loss += self.criterion(output, target).item()
                
        return val_loss / len(val_loader)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              early_stopping_patience: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the model"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                       f'Train Loss: {train_loss:.6f}, '
                       f'Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info(f'New best model saved with validation loss: {best_val_loss:.6f}')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def plot_losses(self, save_path: str = 'training_losses.png'):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def log_training_session(self,
                           symbol: str,
                           start_date: datetime,
                           end_date: datetime,
                           train_loss: float,
                           val_loss: float):
        """Log training session details"""
        record = {
            'timestamp': str(datetime.now()),
            'symbol': symbol,
            'start_date': str(start_date),
            'end_date': str(end_date),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_path': self.model_save_path
        }
        
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                try:
                    history = json.load(f)
                    if isinstance(history, dict):
                        # Convert dict to list (legacy bug)
                        history = [history]
                except Exception:
                    history = []
        else:
            history = []
            
        history.append(record)
        
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def load_model(self, model_path: str) -> None:
        """Load model with version handling"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # New format with metadata
                self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
                logger.info(f"Loaded model from {model_path} with metadata")
            else:
                # Old format, just state dict
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_path} (legacy format)")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise 