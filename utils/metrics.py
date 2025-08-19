import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_regression_metrics(y_true, y_pred):
    """Calculate common regression metrics"""
    y_true = torch.tensor(y_true) if not isinstance(y_true, torch.Tensor) else y_true
    y_pred = torch.tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    
    # R-squared
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def calculate_classification_metrics(y_true, y_pred):
    """Calculate common classification metrics"""
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    accuracy = np.mean(y_true_np == y_pred_np)
    
    return {
        'Accuracy': accuracy,
        'Classification Report': classification_report(y_true_np, y_pred_np),
        'Confusion Matrix': confusion_matrix(y_true_np, y_pred_np)
    }

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
