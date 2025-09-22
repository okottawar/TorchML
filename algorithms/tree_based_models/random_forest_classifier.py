import torch
from ...utils.base import BaseClassifier
from .decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier(BaseClassifier):
    """
    A Random Forest classifier.

    This model fits a number of decision tree classifiers on various sub-samples
    of the dataset (bagging) and uses majority voting to improve predictive 
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
        Build a forest of decision tree classifiers from the training set (X, y).
        """
        self.trees = []
        n_samples = X.shape[0]
        
        X = self._to_tensor(X)
        y = self._to_tensor(y).long()

        for _ in range(self.n_estimators):
            # --- Bagging (Bootstrap Aggregating) ---
            # Create a random subsample of the data with replacement
            idxs = torch.randint(0, n_samples, (n_samples,))
            X_sample, y_sample = X[idxs], y[idxs]

            # Create and train a new decision tree on the sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions for X by aggregating predictions from all trees
        using majority voting.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)

        # Get predictions from each tree in the forest
        all_predictions = [tree.predict(X) for tree in self.trees]
        # Stack predictions into a (n_samples, n_estimators) tensor
        predictions_stack = torch.stack(all_predictions, dim=1)

        # --- Aggregation: Majority Voting ---
        # torch.mode returns the most frequent value (the vote) for each sample
        y_pred, _ = torch.mode(predictions_stack, dim=1)
        return y_pred
