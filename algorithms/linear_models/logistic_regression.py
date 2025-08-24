# algorithms/linear_models/logistic_regression.py

import torch
from ..base import BaseClassifier

class LogisticRegression(BaseClassifier):
    """
    Logistic Regression implemented from scratch in PyTorch.
    Uses gradient descent and autograd to minimize Binary Cross-Entropy loss.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def fit(self, X, y):
        """Fit logistic regression using gradient descent with autograd."""
        X = self._to_tensor(X)
        y = self._to_tensor(y, reshape_y=True)
        n_samples, n_features = X.shape

        self.weights = torch.randn(n_features, 1, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)
        
        print("Training Logistic Regression...")
        for epoch in range(self.max_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred_proba = self._sigmoid(z)

            # Calculate loss (Binary Cross-Entropy)
            epsilon = 1e-15
            y_pred_clipped = torch.clamp(y_pred_proba, epsilon, 1 - epsilon)
            loss = -torch.mean(y * torch.log(y_pred_clipped) + (1 - y) * torch.log(1 - y_pred_clipped))
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
                
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break
        
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        self._check_is_fitted()
        X = self._to_tensor(X)
        with torch.no_grad():
            z = X @ self.weights + self.bias
            return self._sigmoid(z).flatten()

    def predict(self, X):
        """Make binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).float()

    def get_parameters(self):
        """Get learned parameters."""
        self._check_is_fitted()
        return {
            'weights': self.weights.detach().numpy(),
            'bias': self.bias.detach().numpy()
        }