# Regularized Regression (Ridge & Lasso)

## Project Overview

This project implements **Regularized Regression techniques** including **Ridge Regression** and **Lasso Regression** to improve model performance and reduce overfitting in a regression problem.

The dataset used contains housing-related variables where the target variable is **MEDV (Median Value of Homes)**.

The workflow includes:

* Multicollinearity analysis using **Variance Inflation Factor (VIF)**
* Training **Ridge Regression**
* Training **Lasso Regression**
* Finding the best **regularization parameter (lambda / alpha)**
* Model evaluation using **MSE** and **RMSE**

---

# Dataset

The dataset contains several features describing housing conditions.

**Target Variable**

* `medv` → Median value of owner-occupied homes

**Example Features**

* Crime rate
* Number of rooms
* Property tax rate
* Distance to employment centers
* Other housing indicators

---

# Project Workflow

## 1. Multicollinearity Check

Before training the model, **Variance Inflation Factor (VIF)** is calculated to detect multicollinearity between features.

Libraries used:

* `statsmodels`

Purpose:

* Identify highly correlated features
* Ensure regression assumptions are satisfied

---

## 2. Model Training

### Ridge Regression

Ridge Regression is used to reduce overfitting by adding **L2 regularization**.

Formula:

minimize:

Loss = RSS + λ Σ β²

Where:

* RSS = Residual Sum of Squares
* λ = regularization parameter

Implementation:

* `sklearn.linear_model.Ridge`

Outputs:

* Model coefficients
* Intercept
* Feature importance

---

### Lasso Regression

Lasso Regression applies **L1 regularization** which can shrink some coefficients to **zero**, effectively performing feature selection.

Formula:

Loss = RSS + λ Σ |β|

Implementation:

* `sklearn.linear_model.Lasso`

Outputs:

* Feature coefficients
* Selected features

---

## 3. Finding the Best Lambda (Alpha)

Several values of **alpha** were tested:

* 0.01
* 0.1
* 1
* 10

Each model is trained and evaluated using validation data.

Evaluation metrics:

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**

Library used:

* `sklearn.metrics`

---

# Model Evaluation

For each alpha value:

MSE = Mean Squared Error

RMSE = √MSE

Lower RMSE indicates better model performance.

Example output:

Alpha = 0.01 → RMSE = ...
Alpha = 0.1 → RMSE = ...
Alpha = 1 → RMSE = ...
Alpha = 10 → RMSE = ...

The model with the **lowest RMSE** is selected as the best model.

---

# Libraries Used

* pandas
* numpy
* scikit-learn
* statsmodels

---

# Project Structure

```
Regularized-Regression
│
├── dataset.csv
├── regularized_regression.ipynb
├── README.md
└── requirements.txt
```

---

# Key Concepts

* Linear Regression
* Multicollinearity
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization)
* Model Evaluation (MSE, RMSE)

---

# Author

Machine Learning Practice Project
Regularized Regression Implementation using Python and Scikit-Learn
