import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "src/data/raw_placement_data.csv"
FIGURES_DIR = "reports/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

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


# Convert college tier
college_tier_mapping = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3
}

if X["college_tier"].dtype == "object":
    X["college_tier"] = X["college_tier"].map(
        college_tier_mapping
    )


# Convert target
if y.dtype == "object":
    y = y.map({
        "Placed": 1,
        "Not Placed": 0
    })


# ============================================================
# ONE-HOT ENCODING
# ============================================================

categorical_features = ["branch"]

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

feature_names = preprocessor.get_feature_names_out()

X_encoded = pd.DataFrame(
    X_encoded,
    columns=feature_names
)

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

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# XGBOOST
# ============================================================

print("\n========================================")
print("XGBOOST")
print("========================================")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(
    y_test, xgb_pred, zero_division=0
)
xgb_recall = recall_score(
    y_test, xgb_pred, zero_division=0
)
xgb_f1 = f1_score(
    y_test, xgb_pred, zero_division=0
)

print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1 Score:  {xgb_f1:.4f}")


# ============================================================
# LIGHTGBM
# ============================================================

print("\n========================================")
print("LIGHTGBM")
print("========================================")

lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

lgbm_model.fit(X_train, y_train)

lgbm_pred = lgbm_model.predict(X_test)

lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
lgbm_precision = precision_score(
    y_test, lgbm_pred, zero_division=0
)
lgbm_recall = recall_score(
    y_test, lgbm_pred, zero_division=0
)
lgbm_f1 = f1_score(
    y_test, lgbm_pred, zero_division=0
)

print(f"Accuracy:  {lgbm_accuracy:.4f}")
print(f"Precision: {lgbm_precision:.4f}")
print(f"Recall:    {lgbm_recall:.4f}")
print(f"F1 Score:  {lgbm_f1:.4f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n========================================")
print("XGBOOST vs LIGHTGBM")
print("========================================")

comparison = pd.DataFrame({
    "Model": [
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        xgb_accuracy,
        lgbm_accuracy
    ],
    "Precision": [
        xgb_precision,
        lgbm_precision
    ],
    "Recall": [
        xgb_recall,
        lgbm_recall
    ],
    "F1 Score": [
        xgb_f1,
        lgbm_f1
    ]
})

print(comparison.to_string(index=False))

comparison.to_csv(
    "reports/xgb_lgbm_comparison.csv",
    index=False
)

print("-> Saved model comparison CSV")


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("XGBOOST FEATURE IMPORTANCE")
print("========================================")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    "reports/xgb_feature_importance.csv",
    index=False
)

print(importance.to_string(index=False))


# ============================================================
# SHAP EXPLANATION
# ============================================================

print("\n========================================")
print("SHAP ANALYSIS")
print("========================================")

# Use a sample for SHAP to reduce computation time
sample_size = min(2000, len(X_test))

X_shap = X_test.sample(
    sample_size,
    random_state=42
)

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_shap)


# ============================================================
# SHAP SUMMARY BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/shap_summary_bar.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("-> Saved SHAP summary bar plot")


# ============================================================
# SHAP SUMMARY DOT PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("-> Saved SHAP summary plot")


# ============================================================
# SHAP DEPENDENCE PLOT
# ============================================================

# Find the most important feature
top_feature = importance.iloc[0]["Feature"]

shap.dependence_plot(
    top_feature,
    shap_values,
    X_shap,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/shap_dependence.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"Top SHAP feature: {top_feature}")
print("-> Saved SHAP dependence plot")


# ============================================================
# SHAP FEATURE IMPORTANCE REPORT
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_report = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_report = shap_report.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

shap_report.to_csv(
    "reports/shap_feature_importance.csv",
    index=False
)

print("-> Saved SHAP feature importance CSV")


# ============================================================
# COMPLETION
# ============================================================

print("\n========================================")
print("LAB 10 COMPLETED SUCCESSFULLY")
print("========================================")