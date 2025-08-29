# algorithms/linear_models/elastic_net_regression.py

import torch
from ...utils.base import BaseRegressor

class ElasticNetRegression(BaseRegressor):
    """
    Elastic Net Regression (L1 + L2 Regularized) implemented from scratch.
    It combines the penalties of Lasso and Ridge regression. The optimization is
    done using proximal gradient descent to handle the non-differentiable L1 term.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__()
        self.alpha = alpha              # Overall regularization strength
        self.l1_ratio = l1_ratio        # The mix between L1 and L2 (0=Ridge, 1=Lasso)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _soft_thresholding(self, w, lmbd):
        """Proximal operator for the L1 norm (soft-thresholding)."""
        return torch.sign(w) * torch.maximum(torch.abs(w) - lmbd, torch.zeros_like(w))

    def fit(self, X, y):
        """Fit Elastic Net regression using proximal gradient descent."""
        X = self._to_tensor(X)
        y = self._to_tensor(y, reshape_y=True)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = torch.randn(n_features, 1) * 0.01
        self.bias = torch.zeros(1)
        
        # Calculate L1 and L2 strengths based on alpha and l1_ratio
        l1_strength = self.alpha * self.l1_ratio
        l2_strength = self.alpha * (1 - self.l1_ratio)

        print(f"Training Elastic Net with alpha={self.alpha} and l1_ratio={self.l1_ratio}")
        
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Calculate loss for monitoring purposes
            mse_loss = torch.mean((y_pred - y) ** 2)
            l1_penalty = l1_strength * torch.sum(torch.abs(self.weights))
            l2_penalty = l2_strength * torch.sum(self.weights ** 2)
            total_loss = mse_loss + l1_penalty + l2_penalty
            self.loss_history.append(total_loss.item())

            # --- Proximal Gradient Descent Step ---
            # 1. Compute gradient of the differentiable part (MSE + L2 penalty)
            residual = y_pred - y
            grad_w = (2 / n_samples) * X.T @ residual + 2 * l2_strength * self.weights
            grad_b = (2 / n_samples) * torch.sum(residual)
            
            # 2. Perform a standard gradient descent step
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # 3. Apply the proximal operator for the L1 part (soft-thresholding)
            threshold = self.learning_rate * l1_strength
            self.weights = self._soft_thresholding(self.weights, threshold)

            # Check for convergence
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