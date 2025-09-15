#algorithms/tree_based_models/decision_tree_classifier

import torch
from ...utils.base import BaseClassifier 

class DecisionTreeClassifierNode:
    """
    A single node in the Decision Tree. This class handles the recursive splitting for classification.
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
        is_leaf = (
            self.depth >= max_depth or
            len(y) < min_samples_split or
            len(torch.unique(y)) == 1 
        )

        if is_leaf:
            self.value = self._most_common_class(y)
            return

        feature, threshold, left_idx, right_idx = self._best_split(X, y)

        if feature is None:
            self.value = self._most_common_class(y)
            return

        self.feature_index = feature
        self.threshold = threshold

        # Create child nodes and recursively fit them
        self.left = DecisionTreeClassifierNode(depth=self.depth + 1)
        self.left.fit(X[left_idx], y[left_idx], max_depth, min_samples_split)

        self.right = DecisionTreeClassifierNode(depth=self.depth + 1)
        self.right.fit(X[right_idx], y[right_idx], max_depth, min_samples_split)

    def _gini_impurity(self, y):
        """Calculate the Gini impurity for a set of labels."""
        n_samples = len(y)
        if n_samples == 0:
            return 0.0
        
        _, counts = torch.unique(y, return_counts=True)
        probs = counts / n_samples
        gini = 1 - torch.sum(probs**2)
        return gini

    def _best_split(self, X, y):
        """Find the best feature and threshold to split on based on Gini impurity reduction."""
        n_samples, n_features = X.shape
        best_impurity_reduction = -1
        best_feature, best_threshold = None, None
        best_left_idx, best_right_idx = None, None

        parent_impurity = self._gini_impurity(y)
        
        for feature in range(n_features):
            unique_vals = torch.unique(X[:, feature])
            sorted_vals = torch.sort(unique_vals).values
            
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2 if len(sorted_vals) > 1 else sorted_vals

            for threshold in thresholds:
                left_idx = torch.where(X[:, feature] <= threshold)[0]
                right_idx = torch.where(X[:, feature] > threshold)[0]

                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                left_impurity = self._gini_impurity(y[left_idx])
                right_impurity = self._gini_impurity(y[right_idx])
                weighted_impurity = (len(left_idx) / n_samples) * left_impurity + (len(right_idx) / n_samples) * right_impurity
                impurity_reduction = parent_impurity - weighted_impurity

                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        
        return best_feature, best_threshold, best_left_idx, best_right_idx

    def _most_common_class(self, y):
        """Find the most frequent class in a set of labels."""
        if len(y) == 0:
            return None
        vals, counts = torch.unique(y, return_counts=True)
        return vals[torch.argmax(counts)]

    def predict(self, X):
        """Recursively predict the class for a set of inputs."""
        if self.value is not None:
            return self.value.expand(X.shape[0])
        
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        
        y_pred = torch.empty(X.shape[0], dtype=torch.long)

        if torch.any(left_mask):
            y_pred[left_mask] = self.left.predict(X[left_mask])
        if torch.any(right_mask):
            y_pred[right_mask] = self.right.predict(X[right_mask])
            
        return y_pred

class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier from scratch.

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
        """Build the decision tree classifier from training data."""
        X = self._to_tensor(X)
        y = self._to_tensor(y).long() 

        self.root = DecisionTreeClassifierNode(depth=0)
        self.root.fit(X, y, self.max_depth, self.min_samples_split)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions on new data."""
        self._check_is_fitted()
        X = self._to_tensor(X)
        return self.root.predict(X)