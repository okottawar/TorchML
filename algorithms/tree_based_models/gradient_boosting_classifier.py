import torch
from ...utils.base import BaseClassifier
from .decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingClassifier(BaseClassifier):
    """
    Gradient Boosting Classifier from scratch for binary classification.

    This model builds an ensemble of weak learners (regression trees) sequentially.
    Each tree is trained to predict the pseudo-residuals of the previous model,
    based on the gradient of the logistic loss function.

    Args:
        n_estimators (int): The number of boosting stages.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): The maximum depth of the individual regression estimators.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def fit(self, X, y):
        """
        Build the Gradient Boosting model from the training set (X, y).
        """
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        # Step 1: Initialize model with the log of odds
        p = torch.mean(y)
        self.initial_prediction = torch.log(p / (1 - p))
        current_log_odds = self.initial_prediction.expand(len(y))

        for _ in range(self.n_estimators):
            # Convert current log-odds to probabilities
            current_probs = self._sigmoid(current_log_odds)

            # Step 2a: Compute pseudo-residuals
            residuals = y - current_probs

            # Step 2b: Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Step 2c: Update the model's predictions (in log-odds space)
            update = self.learning_rate * tree.predict(X)
            current_log_odds += update

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)

        log_odds = self.initial_prediction.expand(X.shape[0])

        # Add predictions from each tree
        for tree in self.trees:
            log_odds += self.learning_rate * tree.predict(X)

        # Convert final log-odds to probabilities
        return self._sigmoid(log_odds)

    def predict(self, X):
        """
        Predict class labels for X.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).float()
