import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "src/data/raw_placement_data.csv"
FIGURES_DIR = "reports/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print(f"Total samples: {len(df)}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

features = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

target = "placement_status"

X = df[features].copy()
y = df[target].copy()


# Convert college tier to numeric
college_tier_mapping = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3
}

if X["college_tier"].dtype == "object":
    X["college_tier"] = X["college_tier"].map(college_tier_mapping)


# Convert target to binary
if y.dtype == "object":
    y = y.map({
        "Placed": 1,
        "Not Placed": 0
    })

print(f"Number of features before encoding: {X.shape[1]}")


# ============================================================
# ONE-HOT ENCODE BRANCH
# ============================================================

categorical_features = ["branch"]
numeric_features = [
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "branch",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

X_encoded = preprocessor.fit_transform(X)

print(f"Number of features after encoding: {X_encoded.shape[1]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ============================================================
# BASELINE DECISION TREE
# ============================================================

print("\n========================================")
print("BASELINE DECISION TREE")
print("========================================")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)

print(f"Decision Tree Accuracy:  {dt_accuracy:.4f}")
print(f"Decision Tree Precision: {dt_precision:.4f}")
print(f"Decision Tree Recall:    {dt_recall:.4f}")
print(f"Decision Tree F1 Score:  {dt_f1:.4f}")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

rf = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)

print(f"Random Forest Accuracy:  {rf_accuracy:.4f}")
print(f"Random Forest Precision: {rf_precision:.4f}")
print(f"Random Forest Recall:    {rf_recall:.4f}")
print(f"Random Forest F1 Score:  {rf_f1:.4f}")
print(f"Random Forest OOB Score: {rf.oob_score_:.4f}")


# ============================================================
# COMPARE DECISION TREE AND RANDOM FOREST
# ============================================================

print("\n========================================")
print("DECISION TREE vs RANDOM FOREST")
print("========================================")

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        dt_precision,
        rf_precision
    ],
    "Recall": [
        dt_recall,
        rf_recall
    ],
    "F1 Score": [
        dt_f1,
        rf_f1
    ]
})

print(comparison.to_string(index=False))

comparison.to_csv(
    "reports/random_forest_model_comparison.csv",
    index=False
)


# ============================================================
# NUMBER OF TREES
# ============================================================

print("\n========================================")
print("EFFECT OF NUMBER OF TREES")
print("========================================")

tree_numbers = [10, 25, 50, 100, 200]

accuracies = []
oob_errors = []

for n in tree_numbers:

    model = RandomForestClassifier(
        n_estimators=n,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)
    oob_errors.append(1 - model.oob_score_)

    print(
        f"Trees: {n:3d} | "
        f"Accuracy: {accuracy:.4f} | "
        f"OOB Error: {1 - model.oob_score_:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.plot(
    tree_numbers,
    accuracies,
    marker="o",
    label="Test Accuracy"
)

plt.plot(
    tree_numbers,
    oob_errors,
    marker="s",
    label="OOB Error"
)

plt.xlabel("Number of Trees")
plt.ylabel("Score")
plt.title("Effect of Number of Trees")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/random_forest_number_of_trees.png",
    dpi=150
)

plt.close()

print("-> Saved number of trees graph")


# ============================================================
# FEATURE SUBSAMPLING
# ============================================================

print("\n========================================")
print("EFFECT OF FEATURE SUBSAMPLING")
print("========================================")

feature_options = ["sqrt", "log2", None]

feature_labels = [
    "sqrt",
    "log2",
    "None"
]

feature_accuracies = []

for option in feature_options:

    model = RandomForestClassifier(
        n_estimators=100,
        max_features=option,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    feature_accuracies.append(accuracy)

    print(
        f"max_features={option} | "
        f"Accuracy: {accuracy:.4f}"
    )


plt.figure(figsize=(8, 5))

plt.bar(
    feature_labels,
    feature_accuracies
)

plt.xlabel("Feature Subsampling Method")
plt.ylabel("Test Accuracy")
plt.title("Effect of Feature Subsampling")

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/random_forest_feature_subsampling.png",
    dpi=150
)

plt.close()

print("-> Saved feature subsampling graph")


# ============================================================
# COMPLETION
# ============================================================

print("\n========================================")
print("LAB 9 COMPLETED SUCCESSFULLY")
print("========================================")