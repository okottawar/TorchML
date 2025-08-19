"""Base classes and interfaces for all algorithms"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class BaseAlgorithm(ABC):
    """Base interface for all ML algorithms"""
    
    def __init__(self):
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'BaseAlgorithm':
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data"""
        pass
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Return the mean accuracy/R2 score"""
        predictions = self.predict(X)
        if self._is_classifier():
            return (predictions == y).float().mean().item()
        else:
            return self._r2_score(y, predictions)
    
    def _is_classifier(self) -> bool:
        """Check if this is a classification algorithm"""
        return hasattr(self, '_classification') and self._classification
    
    def _r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Calculate R-squared score for regression"""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return (1 - ss_res / ss_tot).item()
    
    def _to_tensor(self, data) -> torch.Tensor:
        """Convert data to PyTorch tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, dtype=torch.float32, device=self.device)

class BaseNeuralNetwork(nn.Module, BaseAlgorithm):
    """Base class for neural network algorithms"""
    
    def __init__(self):
        nn.Module.__init__(self)
        BaseAlgorithm.__init__(self)
        self.optimizer = None
        self.criterion = None
        self.training_history = {'loss': [], 'accuracy': []}
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
            epochs: int = 100, lr: float = 0.01, 
            batch_size: int = 32, verbose: bool = True) -> 'BaseNeuralNetwork':
        """Standard neural network training loop"""
        X, y = self._to_tensor(X), self._to_tensor(y)
        
        # Setup optimizer if not already done
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy for classification
                if self._is_classifier():
                    predicted = (outputs > 0.5).float() if outputs.shape[1] == 1 else outputs.argmax(1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            # Record training metrics
            avg_loss = epoch_loss / len(dataloader)
            self.training_history['loss'].append(avg_loss)
            
            if self._is_classifier():
                accuracy = correct / total
                self.training_history['accuracy'].append(accuracy)
                if verbose and (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            else:
                if verbose and (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        self.is_fitted = True
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        X = self._to_tensor(X)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            if self._is_classifier():
                return (outputs > 0.5).float() if outputs.shape[1] == 1 else outputs.argmax(1)
            return outputs
