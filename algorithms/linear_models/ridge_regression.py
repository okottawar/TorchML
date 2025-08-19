import torch

from ..base import BaseRegressor

class RidgeRegression(BaseRegressor):
    """
    Ridge Regression (L2 Regularized Linear Regression) implemented from scratch in PyTorch.
    Ridge Regression adds an L2 penalty term to the loss function to shrink weights,
    improving model stability and reducing overfitting, especially in the presence of
    multicollinearity.

    Mathematical formulation:
    Loss = MSE + alpha * ||w||_2^2 = (1/m) * ||y - Xw - b||^2 + alpha * sum(w_i^2)
    The L2 penalty is differentiable, allowing optimization via standard gradient descent.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, alpha=0.1):
        """
        Initialize Ridge Regression

        Args:
            learning_rate (float): Step size for gradient descent
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            alpha (float): L2 regularization strength. Higher values = more shrinkage
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha  # L2 regularization parameter
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit Ridge Regression using gradient descent

        The algorithm minimizes the MSE loss plus L2 penalty:
        - Loss: (1/m) * ||y - Xw - b||^2 + alpha * sum(w_i^2)
        - Gradients: dL/dw = (2/m) * X^T (y_hat - y) + 2 alpha w, dL/db = (2/m) * sum(y_hat - y)

        Args:
            X (torch.Tensor or np.ndarray): Training features [n_samples, n_features]
            y (torch.Tensor or np.ndarray): Training targets [n_samples,]

        Returns:
            self: Fitted estimator
        """
        # Convert inputs to PyTorch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = torch.randn(n_features, 1) * 0.01  # Small random initialization
        self.bias = torch.zeros(1)  # Initialize bias to zero

        print(f"Training Ridge with alpha={self.alpha}")

        # Training loop using gradient descent
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Calculate total loss (MSE + L2 penalty)
            mse_loss = torch.mean((y_pred - y) ** 2)
            l2_penalty = self.alpha * torch.sum(self.weights ** 2)
            total_loss = mse_loss + l2_penalty
            self.loss_history.append(total_loss.item())

            # Compute gradients
            residual = y_pred - y
            grad_w = 2 / n_samples * X.T @ residual + 2 * self.alpha * self.weights
            grad_b = 2 / n_samples * torch.sum(residual)

            # Update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Convergence check
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

            # Progress reporting
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}, "
                      f"MSE = {mse_loss:.6f}, "
                      f"L2 = {l2_penalty:.6f}")

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
