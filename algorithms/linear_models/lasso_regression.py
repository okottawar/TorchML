import torch

from ..base import BaseRegressor

class LassoRegression(BaseRegressor):
    """
    Lasso Regression (L1 Regularized Linear Regression) implemented from scratch in PyTorch.
    The Lasso performs automatic feature selection by adding an L1 penalty term that
    encourages sparsity in the weight vector. This makes it particularly useful for
    high-dimensional data where feature selection is important.

    Mathematical formulation:
    Loss = MSE + alpha * ||w||_1 = (1/2n) * ||y - Xw||^2 + alpha * sum(|w_i|)

    The L1 penalty creates a non-differentiable point at w=0, requiring special
    optimization techniques like soft-thresholding (proximal gradient method).
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, alpha=0.1):
        """
        Initialize Lasso Regression

        Args:
            learning_rate (float): Step size for gradient descent
            max_iter (int): Maximum number of iterations
            tol (float): Tolerance for convergence
            alpha (float): L1 regularization strength. Higher values = more sparsity
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha  # L1 regularization parameter
        self.weights = None
        self.bias = None
        self.loss_history = []

    def soft_thresholding(self, w, lmbd):
        """
        Soft-thresholding operator for L1 penalty optimization.
        This is the proximal operator for L1 norm:
        soft_thresh(w, lambda) = sign(w) * max(|w| - lambda, 0)

        Args:
            w (torch.Tensor): Weight vector
            lmbd (float): Threshold parameter (alpha * learning_rate)

        Returns:
            torch.Tensor: Soft-thresholded weights
        """
        return torch.sign(w) * torch.maximum(torch.abs(w) - lmbd, torch.zeros_like(w))

    def fit(self, X, y):
        """
        Fit Lasso regression using proximal gradient descent.
        The algorithm alternates between:
        1. Gradient descent step on the MSE loss
        2. Soft-thresholding step to handle L1 penalty

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

        print(f"Training Lasso with alpha={self.alpha}")

        # Training loop using proximal gradient descent
        for epoch in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Calculate total loss (MSE + L1 penalty)
            mse_loss = torch.mean((y_pred - y) ** 2)
            l1_penalty = self.alpha * torch.sum(torch.abs(self.weights))
            total_loss = mse_loss + l1_penalty
            self.loss_history.append(total_loss.item())

            # Compute gradients manually (since L1 is non-differentiable at 0)
            residual = y_pred - y
            grad_w = 2 / n_samples * X.T @ residual  # MSE gradient w.r.t. weights
            grad_b = 2 / n_samples * torch.sum(residual)  # MSE gradient w.r.t. bias

            # Standard gradient descent step for MSE part
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Soft-thresholding step for L1 penalty (proximal operator)
            threshold = self.alpha * self.learning_rate
            self.weights = self.soft_thresholding(self.weights, threshold)

            # Convergence check
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

            # Progress reporting
            if epoch % 100 == 0:
                sparsity = torch.sum(torch.abs(self.weights) < 1e-6).item()
                print(f"Epoch {epoch}: Loss = {total_loss:.6f}, "
                      f"MSE = {mse_loss:.6f}, "
                      f"L1 = {l1_penalty:.6f}, "
                      f"Sparse features = {sparsity}/{n_features}")

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
            dict: Dictionary containing weights, bias, sparsity ratio, and active features
        """
        self._check_is_fitted()
        return {
            'weights': self.weights.detach().numpy(),
            'bias': self.bias.detach().numpy(),
            'sparsity_ratio': self.get_sparsity_ratio(),
            'active_features': self.get_active_features()
        }