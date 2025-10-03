import torch
from ...utils.base import BaseRegressor
from .decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingRegressor(BaseRegressor):
    """
    Gradient Boosting Regressor from scratch.

    This model builds an additive model in a forward stage-wise fashion. In each
    stage, a regression tree is fit on the negative gradient (residuals) of the
    given loss function.
    
    Args:
        n_estimators (int): The number of boosting stages to perform.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): The maximum depth of the individual regression estimators.
        min_samples_split (int): The minimum number of samples required to split a node.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Build the Gradient Boosting model from the training set (X, y).
        """
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        # Step 1: Initialize model with a constant value (mean of y)
        self.initial_prediction = torch.mean(y)
        current_predictions = self.initial_prediction.expand(len(y))

        # Step 2: Iteratively build trees
        for _ in range(self.n_estimators):
            # Step 2a: Compute pseudo-residuals (gradient of the loss function)
            residuals = y - current_predictions

            # Step 2b: Fit a weak learner (decision tree) to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Step 2c: Update the model's predictions
            update = self.learning_rate * tree.predict(X)
            current_predictions += update
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions with the trained Gradient Boosting model.
        """
        self._check_is_fitted()
        X = self._to_tensor(X)

        # Start with the initial constant prediction
        y_pred = self.initial_prediction.expand(X.shape[0])

        # Add the predictions from each tree in the ensemble
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
