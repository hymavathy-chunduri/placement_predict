import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import (
    make_classification,
    make_blobs
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


os.makedirs(
    "reports/figures",
    exist_ok=True
)


# ==================================================
# PART A: BINARY CLASSIFICATION
# ==================================================

print("\n========================================")
print("PART A: BINARY LOGISTIC REGRESSION")
print("========================================")


# 1. Generate Binary Dataset

X_bin, y_bin = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_classes=2
)


# 2. Train-Test Split

X_bin_train, X_bin_test, y_bin_train, y_bin_test = (
    train_test_split(
        X_bin,
        y_bin,
        test_size=0.20,
        random_state=42
    )
)


# 3. Train Binary Logistic Regression

bin_model = LogisticRegression()

bin_model.fit(
    X_bin_train,
    y_bin_train
)


# 4. Predict and Evaluate

y_bin_pred = bin_model.predict(
    X_bin_test
)


print(
    "\n--- Binary Classification Report ---"
)

print(
    classification_report(
        y_bin_test,
        y_bin_pred
    )
)

binary_accuracy = accuracy_score(
    y_bin_test,
    y_bin_pred
)

print(
    f"Binary Accuracy: "
    f"{binary_accuracy:.4f}"
)


# 5. Binary Decision Boundary

xx, yy = np.meshgrid(
    np.linspace(
        X_bin[:, 0].min() - 1,
        X_bin[:, 0].max() + 1,
        200
    ),

    np.linspace(
        X_bin[:, 1].min() - 1,
        X_bin[:, 1].max() + 1,
        200
    )
)


Z = bin_model.predict(
    np.c_[
        xx.ravel(),
        yy.ravel()
    ]
).reshape(
    xx.shape
)


plt.figure(
    figsize=(8, 6)
)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

plt.scatter(
    X_bin_test[:, 0],
    X_bin_test[:, 1],
    c=y_bin_test,
    edgecolors="k",
    cmap=plt.cm.coolwarm
)

plt.title(
    "Binary Logistic Regression Decision Boundary"
)

plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.tight_layout()


binary_plot = (
    "reports/figures/"
    "binary_logistic_decision_boundary.png"
)

plt.savefig(
    binary_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"-> Saved binary decision boundary to "
    f"{binary_plot}"
)


# ==================================================
# PART B: MULTICLASS CLASSIFICATION
# ==================================================

print("\n========================================")
print("PART B: MULTICLASS LOGISTIC REGRESSION")
print("========================================")


# 1. Generate Multiclass Dataset

X_multi, y_multi = make_blobs(
    n_samples=1500,
    n_features=2,
    centers=3,
    random_state=42
)


# 2. Train-Test Split

X_m_train, X_m_test, y_m_train, y_m_test = (
    train_test_split(
        X_multi,
        y_multi,
        test_size=0.20,
        random_state=42
    )
)


# 3. Multinomial Logistic Regression

multi_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs"
)

multi_model.fit(
    X_m_train,
    y_m_train
)


# 4. One-vs-Rest Model

ovr_model = LogisticRegression(
    multi_class="ovr",
    solver="lbfgs"
)

ovr_model.fit(
    X_m_train,
    y_m_train
)


# 5. Evaluate Multinomial Model

y_m_pred = multi_model.predict(
    X_m_test
)

print(
    "\n--- Multinomial Logistic Regression Report ---"
)

print(
    classification_report(
        y_m_test,
        y_m_pred
    )
)

multiclass_accuracy = accuracy_score(
    y_m_test,
    y_m_pred
)

print(
    f"Multiclass Accuracy: "
    f"{multiclass_accuracy:.4f}"
)


# 6. Confusion Matrix

cm = confusion_matrix(
    y_m_test,
    y_m_pred
)

plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cbar=False
)

plt.title(
    "Confusion Matrix - Multiclass Classification"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()


confusion_plot = (
    "reports/figures/"
    "multiclass_confusion_matrix.png"
)

plt.savefig(
    confusion_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"-> Saved confusion matrix to "
    f"{confusion_plot}"
)


print("\n========================================")
print("LAB 7 COMPLETED SUCCESSFULLY")
print("========================================")
