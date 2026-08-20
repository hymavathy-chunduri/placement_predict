import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():
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
    # 2. Select Features and Target
    # --------------------------------------------------

    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    target_col = "salary_package_lpa"

    # Keep only rows with valid values
    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    # Input features
    X_raw = df_clean[feature_cols].values

    # Target
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(
        f"Loaded {N} data points with "
        f"input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )

    # --------------------------------------------------
    # 3. Create Design Matrix
    # --------------------------------------------------

    # Add a column of ones for the intercept:
    #
    # X_design = [1, cgpa, communication_skill_score]

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    # --------------------------------------------------
    # 4. Standard Least Squares
    # --------------------------------------------------

    # Closed-form solution:
    #
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    XT_y = np.dot(
        X_design.T,
        y
    )

    # Try the standard matrix inverse first
    try:

        XT_X_inv = np.linalg.inv(XT_X)

    except np.linalg.LinAlgError:

        print(
            "Matrix is singular. "
            "Using pseudo-inverse instead."
        )

        XT_X_inv = np.linalg.pinv(XT_X)

    # Calculate optimal weights
    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # --------------------------------------------------
    # 5. Display Model Parameters
    # --------------------------------------------------

    print("\n--- Optimal Model Parameters (Weights & Bias) ---")

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[0]} (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for {feature_cols[1]} (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )

    # --------------------------------------------------
    # 6. Calculate Predictions and Error
    # --------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # Sum of Squared Errors objective
    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # --------------------------------------------------
    # 7. Prepare Output Directory
    # --------------------------------------------------

    output_dir = os.path.join(
        "reports",
        "figures"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # 8. Sample Data for Visualization
    # --------------------------------------------------

    # The regression model was calculated using ALL
    # valid data points.
    #
    # Only the visualization uses a sample to avoid
    # excessive memory usage.

    max_plot_points = 5000

    if N > max_plot_points:

        rng = np.random.default_rng(42)

        sample_indices = rng.choice(
            N,
            size=max_plot_points,
            replace=False
        )

    else:

        sample_indices = np.arange(N)

    X_plot = X_raw[sample_indices]
    y_plot = y[sample_indices]

    print(
        f"Plotting {len(sample_indices)} "
        f"sampled data points out of {N}."
    )

    # --------------------------------------------------
    # 9. Create 3D Regression Visualization
    # --------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Actual data points
    ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        y_plot.ravel(),
        alpha=0.5,
        s=10,
        label="Actual Data Points"
    )

    # Create meshgrid for regression plane

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    # Regression equation:
    #
    # y = w0 + w1*x1 + w2*x2

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    # Regression plane
    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )

    # --------------------------------------------------
    # 10. Labels
    # --------------------------------------------------

    ax.set_xlabel(
        "CGPA (Feature 1)"
    )

    ax.set_ylabel(
        "Communication Skills (Feature 2)"
    )

    ax.set_zlabel(
        "Salary Package LPA (Target)"
    )

    ax.set_title(
        "Linear Regression via Standard Least Squares "
        "(P=1, L=2, M=1)"
    )

    ax.legend()

    plt.tight_layout()

    # --------------------------------------------------
    # 11. Save Figure
    # --------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\n-> Successfully saved 3D regression plot to "
        f"{output_path}"
    )


if __name__ == "__main__":
    train_linear_regression_ls()