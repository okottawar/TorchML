# TorchML

## 🚀 TorchML: Core ML Models from Scratch in PyTorch

TorchML is a **work-in-progress** repository dedicated to implementing fundamental machine learning algorithms—starting with linear models—completely from scratch via PyTorch. The aim is both educational and practical: to help learners deeply understand the intuition and mathematics behind each algorithm, beyond simply calling `sklearn`.

***

## ✔️ Completed: Linear Models Section

### 🟢 Implemented Algorithms

- **Linear Regression:** Closed-form and gradient descent implementations with educational comments.
- **Logistic Regression:** Binary classification with gradient descent and optional regularization.
- **Ridge Regression:** L2-regularization to combat multicollinearity and overfitting.
- **Lasso Regression:** L1-regularization for sparse, interpretable, and feature-selective models.
- **Elastic Regression:** A combination of both L1 and L2 regularization penalties.

All models support training, prediction, and direct access to learned parameters. Each algorithm is built to be readable, extensible, and easy to test or compare.

***

## 📦 Repository Structure

```
TorchML/
├── algorithms/
│   └── linear_models/
│       ├── linear_regression.py
│       ├── polynomial_regression.py
│       ├── ridge_regression.py
│       ├── lasso_regression.py
│       └── logistic_regression.py
├── utils/
│   ├── base.py
│   ├── metrics.py
│   └── visualization.py
├── datasets/
│   └── data_loaders.py
├── tests/
│   └── test_algorithms.py
├── experiments/
│   └── algorithm_comparison.py
├── notebooks/
└── README.md
```

***

## 🎓 Educational Goals

- Demystify the inner working of classic ML algorithms.
- Show how mathematical concepts (loss functions, regularization, optimization) translate directly to PyTorch code.
- Enable learners to compare results, tune hyperparameters, and visualize model behavior.
- Provide a clean foundation for extending algorithms or applying them to real data.

***

## 📈 Example Usage

```python
import torch
from algorithms.linear_models.linear_regression import LinearRegression

# Generate sample data
X = torch.randn(100, 2)
y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.0 + 0.1 * torch.randn(100)

# Train model
model = LinearRegression(method='gradient_descent')
model.fit(X, y)

# Predict
y_pred = model.predict(X)
print("R^2 Score:", model.score(X, y))
print("Learned weights:", model.get_coefficients())
```

***

## 📝 Next Steps (WIP)

Planned sections (coming soon):

- Utilities, Datasets, Tests, etc.
- Tree-based models (Decision Trees, Random Forest)
- Neural networks (MLP, CNN, RNN)
- Clustering and unsupervised models
- End-to-end classification/regression workflows
- Model comparisons and visualization tools

***

## ⚡️ Status

**Linear models section is complete. Other sections are in development. Stay tuned!**

## 📄 License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.