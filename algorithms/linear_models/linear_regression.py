import torch
import torch.nn as nn
from ..base import BaseRegressor

class LinearRegression(BaseRegressor):
    """Linear Regression implemented from scratch in PyTorch"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Fit linear regression using gradient descent
        
        Mathematical background:
        - Model: ŷ = Xw + b
        - Loss: L = (1/2n) * ||y - ŷ||²
        - Gradients: ∂L/∂w = X^T(ŷ - y)/n, ∂L/∂b = sum(ŷ - y)/n
        """
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
            
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = torch.randn(n_features, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        
        # Training loop
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Calculate loss (MSE)
            loss = torch.mean((y_pred - y) ** 2)
            self.loss_history.append(loss.item())
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad
                
                # Zero gradients
                self.weights.grad.zero_()
                self.bias.grad.zero_()
            
            # Check convergence
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
                
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        self._check_is_fitted()
        
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            
        with torch.no_grad():
            return X @ self.weights + self.bias
    
    def get_parameters(self):
        """Get learned parameters"""
        self._check_is_fitted()
        return {
            'weights': self.weights.detach().numpy(),
            'bias': self.bias.detach().numpy()
        }
    

import torch

from ..base import BaseRegressor

class LinearRegression(BaseRegressor):
    """
    Linear Regression implemented from scratch in PyTorch.
    Uses gradient descent optimization to find optimal weights and bias
    that minimize the mean squared error loss function.

    Mathematical formulation:
    Model: y_hat = Xw + b
    Loss: L = (1/2n) * ||y - y_hat||^2
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Initialize Linear Regression

        Args:
            learning_rate (float): Step size for gradient descent
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit linear regression using gradient descent

        Mathematical background:
        - Model: y_hat = Xw + b
        - Loss: L = (1/2n) * ||y - y_hat||^2  
        - Gradients: dL/dw = X^T(y_hat - y)/n, dL/db = sum(y_hat - y)/n

        Args:
            X (torch.Tensor or np.ndarray): Training features [n_samples, n_features]
            y (torch.Tensor or np.ndarray): Training targets [n_samples,]

        Returns:
            self: Fitted estimator
        """
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = torch.randn(n_features, 1) * 0.01
        self.bias = torch.zeros(1)

        print("Training Linear Regression...")

        # Training loop
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Calculate loss (MSE)
            loss = torch.mean((y_pred - y) ** 2)
            self.loss_history.append(loss.item())

            # Compute gradients
            residual = y_pred - y
            grad_w = 2 / n_samples * X.T @ residual
            grad_b = 2 / n_samples * torch.sum(residual)

            # Update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Check convergence
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

            # Progress reporting
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the fitted model

        Args:
            X (torch.Tensor or np.ndarray): Input features

        Returns:
            torch.Tensor: Predictions
        """
        self._check_is_fitted()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        return (X @ self.weights + self.bias).flatten()

    def get_parameters(self):
        """
        Get learned parameters

        Returns:
            dict: Dictionary containing weights and bias
        """
        self._check_is_fitted()
        return {
            'weights': self.weights.detach().numpy(),
            'bias': self.bias.detach().numpy()
        }
