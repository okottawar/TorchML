import torch
from ...utils.base import BaseRegressor
from .decision_tree_regressor import DecisionTreeRegressor

class RandomForestRegressor(BaseRegressor):
    """
    A Random Forest regressor.

    This model fits a number of decision tree regressors on various sub-samples
    of the dataset (bagging) and uses averaging to improve the predictive 
    accuracy and control over-fitting.
    
    Args:
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of each tree.
        min_samples_split (int): The minimum number of samples required to split a node.
    """
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """
        Build a forest of decision tree regressors from the training set (X, y).
        """
        self.trees = []
        n_samples = X.shape[0]
        
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        for _ in range(self.n_estimators):
            # --- Bagging (Bootstrap Aggregating) ---
            # Create a random subsample of the data with replacement
            idxs = torch.randint(0, n_samples, (n_samples,))
            X_sample, y_sample = X[idxs], y[idxs]

            # Create and train a new decision tree on the sample
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions for X by averaging the predictions from all trees.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)

        # Get predictions from each tree in the forest
        all_predictions = [tree.predict(X) for tree in self.trees]
        # Stack predictions into a (n_samples, n_estimators) tensor
        predictions_stack = torch.stack(all_predictions, dim=1)

        # --- Aggregation: Averaging ---
        # torch.mean calculates the average prediction for each sample
        return torch.mean(predictions_stack, dim=1)
