# algorithms/linear_models/ridge_regression.py

import torch
from ..base import BaseRegressor

class RidgeRegression(BaseRegressor):
    """
    Ridge Regression (L2 Regularized) implemented from scratch in PyTorch.
    Uses gradient descent and autograd to minimize MSE + L2 penalty.
    """

    def __init__(self, alpha=0.1, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__()
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Fit Ridge regression using gradient descent with autograd."""
        X = self._to_tensor(X)
        y = self._to_tensor(y, reshape_y=True)
        n_samples, n_features = X.shape

        # Initialize parameters with requires_grad=True for autograd
        self.weights = torch.randn(n_features, 1, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        print(f"Training Ridge with alpha={self.alpha}")
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Calculate loss (MSE + L2 penalty)
            mse_loss = torch.mean((y_pred - y) ** 2)
            l2_penalty = self.alpha * torch.sum(self.weights ** 2)
            total_loss = mse_loss + l2_penalty
            self.loss_history.append(total_loss.item())
            
            # Backward pass (compute gradients)
            total_loss.backward()
            
            # Update parameters
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad
                
                # Zero gradients
                self.weights.grad.zero_()
                self.bias.grad.zero_()
            
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions using the fitted model."""
        self._check_is_fitted()
        X = self._to_tensor(X)
        with torch.no_grad():
            return (X @ self.weights + self.bias).flatten()

    def get_parameters(self):
        """Get learned parameters."""
        self._check_is_fitted()
        return {
            'weights': self.weights.detach().numpy(),
            'bias': self.bias.detach().numpy()
        }