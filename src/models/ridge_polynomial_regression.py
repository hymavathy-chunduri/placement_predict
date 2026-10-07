import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 1. Generate Data
np.random.seed(42)

X = np.sort(
    6 * np.random.rand(100, 1) + 4,
    axis=0
)

y = (
    np.sin(X).ravel()
    + np.random.normal(0, 0.2, X.shape[0])
)


# 2. 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 3. Higher-Order Polynomial Features
poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(
    X_train
)

X_test_poly = poly.transform(
    X_test
)


# 4. Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_poly
)

X_test_scaled = scaler.transform(
    X_test_poly
)


# 5. Range of Regularization Parameters
lambdas = np.logspace(
    -4,
    4,
    200
)

train_errors = []
test_errors = []


# 6. Train Ridge Regression for Each Lambda
for lam in lambdas:

    ridge = Ridge(
        alpha=lam
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    y_train_pred = ridge.predict(
        X_train_scaled
    )

    y_test_pred = ridge.predict(
        X_test_scaled
    )

    train_errors.append(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_errors.append(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


# 7. Plot Error Curves
os.makedirs(
    "reports/figures",
    exist_ok=True
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    lambdas,
    train_errors,
    label="Training Error (Ew)",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error (Ew)",
    linewidth=2,
    linestyle="--"
)

plt.xscale(
    "log"
)

plt.xlabel(
    "Regularization Parameter (lambda / alpha)"
)

plt.ylabel(
    "Mean Squared Error (Ew)"
)

plt.title(
    "Regularization Path: Ridge Regression "
    "Overfitting Control (Degree 15)"
)

plt.legend()

plt.grid(
    True,
    which="both",
    linestyle="--"
)

plt.tight_layout()


output_path = (
    "reports/figures/"
    "ridge_regularization_error_curve.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("Lab 6 completed successfully.")

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)

print(
    f"Polynomial degree: {poly_degree}"
)

print(
    f"Number of lambda values tested: "
    f"{len(lambdas)}"
)

print(
    f"-> Saved Ridge regularization graph to "
    f"{output_path}"
)