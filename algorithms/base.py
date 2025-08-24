# utils/base.py

from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseAlgorithm(ABC):
    """Base interface for all ML algorithms."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    def _check_is_fitted(self):
        """Check if the estimator has been fitted."""
        if not self.is_fitted:
            raise AttributeError("This model instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    def _to_tensor(self, data, reshape_y=False):
        """Convert data to a PyTorch float tensor and optionally reshape y."""
        if not isinstance(data, torch.Tensor):
            data = torch.FloatTensor(data)
        if reshape_y:
            return data.reshape(-1, 1)
        return data

class BaseRegressor(BaseAlgorithm):
    """Base class for all regression models."""
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)
        y = self._to_tensor(y)
        
        y_pred = self.predict(X)
        ss_res = torch.sum((y - y_pred) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()

class BaseClassifier(BaseAlgorithm):
    """Base class for all classification models."""
    _classification = True

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        y_pred = self.predict(X)
        return (y_pred == y).float().mean().item()