import torch

from ..base import BaseClassifier

class LogisticRegression(BaseClassifier):
    """
    Logistic Regression implemented from scratch in PyTorch.
    Uses gradient descent with sigmoid activation for binary classification.
    Optimizes the cross-entropy (log-likelihood) loss function.

    Mathematical formulation:
    Model: p = sigmoid(Xw + b) where sigmoid(z) = 1/(1 + exp(-z))
    Loss: L = -[y*log(p) + (1-y)*log(1-p)] (cross-entropy)
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Initialize Logistic Regression

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

    def sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability

        Args:
            z (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Sigmoid output
        """
        # Clip z to prevent overflow
        z = torch.clamp(z, -250, 250)
        return 1 / (1 + torch.exp(-z))

    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent

        Mathematical background:
        - Model: p = sigmoid(Xw + b)
        - Loss: L = -[y*log(p) + (1-y)*log(1-p)] (cross-entropy)
        - Gradients: dL/dw = X^T(p - y)/n, dL/db = sum(p - y)/n

        Args:
            X (torch.Tensor or np.ndarray): Training features [n_samples, n_features]
            y (torch.Tensor or np.ndarray): Training targets [n_samples,] (0 or 1)

        Returns:
            self: Fitted estimator
        """
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = torch.randn(n_features, 1) * 0.01
        self.bias = torch.zeros(1)

        print("Training Logistic Regression...")

        # Training loop
        for epoch in range(self.max_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Calculate loss (cross-entropy)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            y_pred_clipped = torch.clamp(y_pred, epsilon, 1 - epsilon)
            loss = -torch.mean(y * torch.log(y_pred_clipped) + (1 - y) * torch.log(1 - y_pred_clipped))
            self.loss_history.append(loss.item())

            # Compute gradients
            residual = y_pred - y
            grad_w = 1 / n_samples * X.T @ residual
            grad_b = 1 / n_samples * torch.sum(residual)

            # Update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Check convergence
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

            # Progress reporting
            if epoch % 100 == 0:
                accuracy = torch.mean(((y_pred > 0.5).float() == y).float())
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X (torch.Tensor or np.ndarray): Input features

        Returns:
            torch.Tensor: Predicted probabilities
        """
        self._check_is_fitted()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        z = X @ self.weights + self.bias
        return self.sigmoid(z).flatten()

    def predict(self, X):
        """
        Make binary predictions

        Args:
            X (torch.Tensor or np.ndarray): Input features

        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).float()

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
