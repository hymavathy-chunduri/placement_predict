import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src.data.ingest import load_and_validate_data


def compute_cost(X, y, w):
    """
    Computes the cost function:
    
    J(w) = 1/(2m) * sum((Xw - y)^2)
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def gradient_descent(X, y, w, alpha, num_iters):
    """
    Implements Gradient Descent from scratch using NumPy.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # Calculate predictions
        predictions = np.dot(X, w)

        # Calculate errors
        errors = predictions - y

        # Gradient:
        # (1/m) * X^T * (Xw - y)
        gradient = (1 / m) * np.dot(X.T, errors)

        # Update weights
        w = w - alpha * gradient

        # Calculate and store cost
        cost = compute_cost(X, y, w)

        cost_history.append(cost)

    return w, cost_history


def run_gradient_descent_experiment():

    # --------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(DATA_PATH)

    # --------------------------------------------------
    # 2. Select Feature and Target
    # --------------------------------------------------

    feature_cols = ["cgpa"]

    target_col = "salary_package_lpa"

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    X_raw = df_clean[feature_cols].values

    y_raw = df_clean[target_col].values.reshape(-1, 1)

    print("\n========================================")
    print("GRADIENT DESCENT LINEAR REGRESSION")
    print("========================================")

    print(f"Total samples: {len(X_raw)}")

    print(f"Input dimension: {X_raw.shape[1]}")

    print(f"Output dimension: {y_raw.shape[1]}")

    # --------------------------------------------------
    # 3. 80/20 Train-Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print("\n--- Train-Test Split ---")

    print(f"Training samples: {len(X_train)}")

    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------
    # 4. Feature Scaling
    # --------------------------------------------------

    scaler_x = StandardScaler()

    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)

    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)

    y_test_scaled = scaler_y.transform(y_test)

    # --------------------------------------------------
    # 5. Add Intercept Column
    # --------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones((X_train_scaled.shape[0], 1)),
            X_train_scaled
        ]
    )

    X_test_design = np.hstack(
        [
            np.ones((X_test_scaled.shape[0], 1)),
            X_test_scaled
        ]
    )

    # --------------------------------------------------
    # 6. Learning Rate Experiment
    # --------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    results = {}

    plt.figure(figsize=(10, 6))

    print("\n--- Learning Rate Experiment ---")

    for alpha in learning_rates:

        # Initialize weights to zero
        w_init = np.zeros(
            (X_train_design.shape[1], 1)
        )

        # Run gradient descent
        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        # Plot cost history
        plt.plot(
            cost_history,
            label=f"Alpha = {alpha}"
        )

        print(
            f"Alpha = {alpha} | "
            f"Initial Cost = {cost_history[0]:.6f} | "
            f"Final Cost = {cost_history[-1]:.6f}"
        )

    plt.xlabel("Iterations")

    plt.ylabel("Cost Function")

    plt.title(
        "Effect of Learning Rate on "
        "Gradient Descent Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n-> Saved learning rate graph to "
        f"{output_path}"
    )

    # --------------------------------------------------
    # 7. Select Best Learning Rate
    # --------------------------------------------------

    best_alpha = 0.1

    final_w = results[best_alpha]["weights"]

    print("\n========================================")

    print(
        f"--- CUSTOM GRADIENT DESCENT "
        f"(alpha = {best_alpha}) ---"
    )

    print("========================================")

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.4f}"
    )

    print(
        f"Coefficient (w1): "
        f"{final_w[1, 0]:.4f}"
    )

    # --------------------------------------------------
    # 8. Test Custom Gradient Descent
    # --------------------------------------------------

    custom_predictions_scaled = np.dot(
        X_test_design,
        final_w
    )

    custom_predictions = scaler_y.inverse_transform(
        custom_predictions_scaled
    )

    custom_mse = np.mean(
        (custom_predictions - y_test) ** 2
    )

    print(
        f"Custom Gradient Descent "
        f"Test MSE: {custom_mse:.4f}"
    )

    # --------------------------------------------------
    # 9. Scikit-learn Comparison
    # --------------------------------------------------

    sklearn_model = SklearnLinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n========================================")

    print("--- SCIKIT-LEARN COMPARISON ---")

    print("========================================")

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.4f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.4f}"
    )

    # --------------------------------------------------
    # 10. Scikit-learn Test Prediction
    # --------------------------------------------------

    sklearn_predictions_scaled = (
        sklearn_model.predict(X_test_scaled)
    )

    sklearn_predictions = scaler_y.inverse_transform(
        sklearn_predictions_scaled
    )

    sklearn_mse = np.mean(
        (sklearn_predictions - y_test) ** 2
    )

    print(
        f"Scikit-learn Test MSE: "
        f"{sklearn_mse:.4f}"
    )

    # --------------------------------------------------
    # 11. Parameter Comparison
    # --------------------------------------------------

    print("\n========================================")

    print("--- PARAMETER COMPARISON ---")

    print("========================================")

    print(
        f"Custom GD Intercept:     "
        f"{final_w[0, 0]:.4f}"
    )

    print(
        f"Sklearn Intercept:       "
        f"{sklearn_model.intercept_[0]:.4f}"
    )

    print(
        f"Custom GD Coefficient:   "
        f"{final_w[1, 0]:.4f}"
    )

    print(
        f"Sklearn Coefficient:     "
        f"{sklearn_model.coef_[0, 0]:.4f}"
    )

    print("\n========================================")

    print("LAB 5 COMPLETED SUCCESSFULLY")

    print("========================================")


if __name__ == "__main__":
    run_gradient_descent_experiment()