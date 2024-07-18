
# Hyperparameter-Optimization

This repository contains code and documentation for hyperparameter tuning of machine learning models. It provides various strategies for tuning hyperparameters to optimize model performance.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Example](#example)
- [Contributing](#contributing)

## Introduction

Hyperparameter tuning is a crucial step in the machine learning pipeline. Properly tuned hyperparameters can significantly improve the performance of a model. There are different techniques and tools for hyperparameter tuning, including grid search, random search, and Bayesian optimization. This repository offers hyperparameter tuning using grid search.

## Installation

Clone the repository and install the required dependencies:

```bash
cd hyperparameter-tuning
git clone https://github.com/Prasun6736/Hyperparameter-Optimization.git
```
## Usage

This section provides a detailed example of how to use Grid Search for hyperparameter tuning with a `RandomForestClassifier` on a dataset.

### Example

In this example, we'll use Grid Search to tune the hyperparameters of a `RandomForestClassifier` on a dataset.

```python
# Necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = set path
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test Score: {test_score}")
```
## Contributing

Contributions to this repository are welcome! To contribute, follow these steps:

1. **Fork** the repository on GitHub.
2. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
git commit -am 'Add some feature'
git push origin feature/your-feature-name
```
3.Submit a pull request to the main branch of the original repository.




