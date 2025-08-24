# algorithms/linear_models/lasso_regression.py

import torch
from ..base import BaseRegressor

class LassoRegression(BaseRegressor):
    """
    Lasso Regression (L1 Regularized) implemented from scratch in PyTorch.
    Uses proximal gradient descent to handle the non-differentiable L1 penalty.
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

    def _soft_thresholding(self, w, lmbd):
        """Proximal operator for L1 norm."""
        return torch.sign(w) * torch.maximum(torch.abs(w) - lmbd, torch.zeros_like(w))

    def fit(self, X, y):
        """Fit Lasso regression using proximal gradient descent."""
        X = self._to_tensor(X)
        y = self._to_tensor(y, reshape_y=True)
        n_samples, n_features = X.shape

        self.weights = torch.randn(n_features, 1) * 0.01
        self.bias = torch.zeros(1)
        
        print(f"Training Lasso with alpha={self.alpha}")
        for epoch in range(self.max_iter):
            y_pred = X @ self.weights + self.bias
            
            # Calculate loss for monitoring
            mse_loss = torch.mean((y_pred - y) ** 2)
            l1_penalty = self.alpha * torch.sum(torch.abs(self.weights))
            total_loss = mse_loss + l1_penalty
            self.loss_history.append(total_loss.item())

            # Manual gradient of MSE part
            residual = y_pred - y
            grad_w = (2 / n_samples) * X.T @ residual
            grad_b = (2 / n_samples) * torch.sum(residual)

            # Update parameters (gradient step)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Apply proximal operator (soft-thresholding step)
            threshold = self.alpha * self.learning_rate
            self.weights = self._soft_thresholding(self.weights, threshold)

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