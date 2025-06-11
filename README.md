# Boston-Housing-Price-Prediction-A-Comparative-Study-of-Regression-Models

# Boston Housing Price Prediction: Linear Regression Experiment

This document outlines a practical experiment demonstrating the application of Linear Regression to the Boston Housing dataset for price prediction. The experiment focuses on data loading, model training, prediction, and basic evaluation.

## Overview

The primary goal of this experiment is to predict housing prices in the Boston area using a Linear Regression model. The provided code snippet showcases the essential steps of loading the dataset, initializing and training a linear regression model, and then evaluating its performance using the Mean Squared Error (MSE).

## Dataset

The experiment utilizes the Boston Housing dataset, which is a classic dataset for regression tasks. It is typically loaded directly from `sklearn.datasets`. The dataset contains various features describing residential homes in the Boston area, with the target variable being the median value of owner-occupied homes (MEDV).

## Steps

The experiment, as indicated by the screenshot, follows these key steps:

1.  **Import Dependencies**: Imports necessary libraries, including:
    * `numpy` for numerical operations.
    * `matplotlib.pyplot` for plotting and visualization.
    * `load_boston` from `sklearn.datasets` to load the dataset.
    * `LinearRegression` from `sklearn.linear_model` for the regression model.
    * `train_test_split` from `sklearn.model_selection` to split data into training and testing sets.
    * `mean_squared_error` from `sklearn.metrics` for model evaluation.

2.  **Load the dataset**: The Boston housing dataset is loaded using `load_boston()`. The features are assigned to `X` and the target (prices) to `y`.

3.  **Split the dataset**: The dataset is divided into training and testing sets using `train_test_split` with a `test_size` of 0.2 and a `random_state` of 42 for reproducibility.

4.  **Initialize and Train the Model**:
    * A `LinearRegression` model is instantiated.
    * The model is trained using the `fit` method on the training features (`X_train`) and training target (`y_train`).

5.  **Make Predictions**:
    * Predictions are generated on the test set features (`X_test`) using the `predict` method.

6.  **Evaluate the Model**:
    * The `mean_squared_error` is calculated between the actual test labels (`y_test`) and the predicted values (`y_pred`) to assess the model's performance.
    * The calculated MSE is then printed.

## How to Run

1.  **Dependencies**: Ensure you have a Python environment with `numpy`, `matplotlib`, and `scikit-learn` installed.
    You can install them via pip if you haven't already:
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
2.  **Code**: Copy the Python code provided in the screenshot into a `.py` file or a Jupyter Notebook cell.
3.  **Execute**: Run the Python script or the Jupyter Notebook cell. The output will display the Mean Squared Error of the Linear Regression model on the test set.

## Code Snippet (from screenshot)

```python
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston # For loading the dataset
from sklearn.linear_model import LinearRegression # For Linear Regression model
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.metrics import mean_squared_error # For evaluating the model

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data # Features
y = boston.target # Target (housing prices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_reg_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
