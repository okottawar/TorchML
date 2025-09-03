import torch
from ...utils.base import BaseRegressor 

class DecisionTreeRegressorNode:
    """
    A single node in the Decision Tree. This class handles the recursive splitting.
    """
    def __init__(self, depth=0):
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None 

    def fit(self, X, y, max_depth, min_samples_split):
        """Recursively build the tree from this node down."""
        # Check stopping criteria to make this node a leaf
        is_leaf = (
            self.depth >= max_depth or
            len(y) < min_samples_split or
            len(torch.unique(y)) == 1 # All target values are the same
        )

        if is_leaf:
            self.value = torch.mean(y)
            return

        # Find the best split for the current data
        feature, threshold, left_idx, right_idx = self._best_split(X, y, min_samples_split)

        # If no beneficial split is found, make this node a leaf
        if feature is None:
            self.value = torch.mean(y)
            return

        # Store the split information
        self.feature_index = feature
        self.threshold = threshold

        # Create child nodes and recursively fit them
        self.left = DecisionTreeRegressorNode(depth=self.depth + 1)
        self.left.fit(X[left_idx], y[left_idx], max_depth, min_samples_split)

        self.right = DecisionTreeRegressorNode(depth=self.depth + 1)
        self.right.fit(X[right_idx], y[right_idx], max_depth, min_samples_split)

    def _best_split(self, X, y, min_samples_split):
        """Find the best feature and threshold to split the data on."""
        n_samples, n_features = X.shape
        best_variance_reduction = -1
        best_feature, best_threshold = None, None
        best_left_idx, best_right_idx = None, None

        parent_variance = torch.var(y)

        for feature in range(n_features):
            # Instead of checking every unique value, we check midpoints between sorted unique values. This is much more efficient.
            unique_vals = torch.unique(X[:, feature])
            sorted_vals = torch.sort(unique_vals).values
            
            if len(sorted_vals) > 1:
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
            else:
                thresholds = sorted_vals

            for threshold in thresholds:
                left_idx = torch.where(X[:, feature] <= threshold)[0]
                right_idx = torch.where(X[:, feature] > threshold)[0]

                # Ensure the split creates children with enough samples
                if len(left_idx) < min_samples_split // 2 or len(right_idx) < min_samples_split // 2:
                    continue

                # Calculate variance reduction
                left_var = torch.var(y[left_idx])
                right_var = torch.var(y[right_idx])
                weighted_var = (len(left_idx) / n_samples) * left_var + (len(right_idx) / n_samples) * right_var
                variance_reduction = parent_variance - weighted_var

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature = feature
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        
        return best_feature, best_threshold, best_left_idx, best_right_idx

    def predict(self, X):
        """Recursively predict the output for a set of inputs."""
        # If this is a leaf node, return its value for all inputs
        if self.value is not None:
            return self.value.expand(X.shape[0])
        
        # If not a leaf, route data to the correct child node
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        
        y_pred = torch.empty(X.shape[0], dtype=torch.float32)

        # Recursively call predict on children and fill the predictions
        if torch.any(left_mask):
            y_pred[left_mask] = self.left.predict(X[left_mask])
        if torch.any(right_mask):
            y_pred[right_mask] = self.right.predict(X[right_mask])
            
        return y_pred

class DecisionTreeRegressor(BaseRegressor):
    """
    Decision Tree Regressor from scratch.

    Args:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split a node.
    """
    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Build the decision tree from training data."""
        # Use helpers from the base class for consistency
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        self.root = DecisionTreeRegressorNode(depth=0)
        self.root.fit(X, y, self.max_depth, self.min_samples_split)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions on new data."""
        self._check_is_fitted()
        X = self._to_tensor(X)
        return self.root.predict(X)