import torch
import numpy as np
from sklearn.datasets import make_regression, make_classification, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_synthetic_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42):
    """Generate synthetic regression data"""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=random_state)
    return torch.FloatTensor(X), torch.FloatTensor(y)

def load_synthetic_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42):
    """Generate synthetic classification data"""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_classes=n_classes, n_redundant=0, random_state=random_state)
    return torch.FloatTensor(X), torch.LongTensor(y)

def load_real_datasets():
    """Load real-world datasets"""
    datasets = {}
    
    # Iris classification
    iris = load_iris()
    datasets['iris'] = {
        'X': torch.FloatTensor(iris.data),
        'y': torch.LongTensor(iris.target),
        'type': 'classification',
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }
    
    # Diabetes regression
    diabetes = load_diabetes()
    datasets['diabetes'] = {
        'X': torch.FloatTensor(diabetes.data),
        'y': torch.FloatTensor(diabetes.target),
        'type': 'regression',
        'feature_names': diabetes.feature_names
    }
    
    return datasets

def train_test_split_torch(X, y, test_size=0.2, random_state=42):
    """Split PyTorch tensors into train/test sets"""
    X_np = X.numpy()
    y_np = y.numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
            torch.tensor(y_train), torch.tensor(y_test))
