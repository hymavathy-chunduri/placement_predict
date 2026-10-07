from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"

OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


# ---------------------------------------------------------
# LOAD AND PREPARE DATA
# ---------------------------------------------------------

def load():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier such as
    # "Tier 1" -> 1
    # "Tier 2" -> 2
    # "Tier 3" -> 3

    if not pd.api.types.is_numeric_dtype(df["college_tier"]):

        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # One-hot encode branch

    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=float
    )

    # Process target variable

    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError("placement_status must be binary")

        y = y.astype(str).map({
            labels[0]: 0,
            labels[1]: 1
        })

    return X, y


# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------

def run():

    X, y = load()

    print("Total samples:", len(X))
    print("Number of features after encoding:", X.shape[1])

    # 80% training and 20% testing

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Training samples:", len(Xtr))
    print("Testing samples:", len(Xte))


    # -----------------------------------------------------
    # 1. GINI AND ENTROPY
    # -----------------------------------------------------

    print("\n========================================")
    print("GINI vs ENTROPY")
    print("========================================")

    for name, criterion in [
        ("gini", "gini"),
        ("entropy", "entropy")
    ]:

        model = DecisionTreeClassifier(
            criterion=criterion,
            random_state=42
        )

        model.fit(Xtr, ytr)

        predictions = model.predict(Xte)

        accuracy = accuracy_score(
            yte,
            predictions
        )

        print(
            f"{name.title()} test accuracy:",
            round(accuracy, 4)
        )

        # Plot tree

        plt.figure(figsize=(18, 10))

        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=[
                str(x)
                for x in sorted(y.unique())
            ],
            filled=True,
            max_depth=4,
            fontsize=7
        )

        plt.title(
            f"Placement Decision Tree - {name.title()}"
        )

        plt.tight_layout()

        plt.savefig(
            OUT / f"decision_tree_{name}.png",
            dpi=150
        )

        plt.close()

        print(
            f"-> Saved {name} decision tree"
        )


    # -----------------------------------------------------
    # 2. COST COMPLEXITY PRUNING
    # -----------------------------------------------------

    print("\n========================================")
    print("COST COMPLEXITY PRUNING")
    print("========================================")

    path = DecisionTreeClassifier(
        random_state=42
    ).cost_complexity_pruning_path(
        Xtr,
        ytr
    )

    rows = []

    for alpha in path.ccp_alphas:

        model = DecisionTreeClassifier(
            ccp_alpha=alpha,
            random_state=42
        )

        model.fit(Xtr, ytr)

        train_accuracy = accuracy_score(
            ytr,
            model.predict(Xtr)
        )

        test_accuracy = accuracy_score(
            yte,
            model.predict(Xte)
        )

        rows.append([
            alpha,
            train_accuracy,
            test_accuracy,
            model.tree_.node_count
        ])

    pruning_results = pd.DataFrame(
        rows,
        columns=[
            "alpha",
            "train_accuracy",
            "test_accuracy",
            "nodes"
        ]
    )

    best = pruning_results.loc[
        pruning_results["test_accuracy"].idxmax()
    ]

    print(
        "Best CCP alpha:",
        best["alpha"]
    )

    print(
        "Pruned test accuracy:",
        best["test_accuracy"]
    )

    # Plot pruning results

    plt.figure(figsize=(9, 6))

    plt.plot(
        pruning_results["alpha"],
        pruning_results["train_accuracy"],
        label="Training"
    )

    plt.plot(
        pruning_results["alpha"],
        pruning_results["test_accuracy"],
        label="Testing"
    )

    plt.xlabel("CCP Alpha")
    plt.ylabel("Accuracy")

    plt.title(
        "Cost Complexity Pruning - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_ccp.png",
        dpi=150
    )

    plt.close()

    print(
        "-> Saved pruning graph"
    )


    # -----------------------------------------------------
    # 3. EFFECT OF TREE DEPTH
    # -----------------------------------------------------

    print("\n========================================")
    print("EFFECT OF TREE DEPTH")
    print("========================================")

    depths = range(1, 16)

    train_scores = []
    test_scores = []

    for depth in depths:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(Xtr, ytr)

        train_scores.append(
            accuracy_score(
                ytr,
                model.predict(Xtr)
            )
        )

        test_scores.append(
            accuracy_score(
                yte,
                model.predict(Xte)
            )
        )

    best_depth = list(depths)[
        int(np.argmax(test_scores))
    ]

    print(
        "Best depth:",
        best_depth
    )

    print(
        "Best test accuracy:",
        max(test_scores)
    )

    # Plot depth results

    plt.figure(figsize=(9, 6))

    plt.plot(
        depths,
        train_scores,
        marker="o",
        label="Training"
    )

    plt.plot(
        depths,
        test_scores,
        marker="o",
        label="Testing"
    )

    plt.xlabel("Maximum Depth")
    plt.ylabel("Accuracy")

    plt.title(
        "Effect of Tree Depth - Placement Prediction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "decision_tree_depth.png",
        dpi=150
    )

    plt.close()

    print(
        "-> Saved tree depth graph"
    )


    print("\n========================================")
    print("LAB 8 COMPLETED SUCCESSFULLY")
    print("========================================")


# ---------------------------------------------------------
# PROGRAM START
# ---------------------------------------------------------

if __name__ == "__main__":
    run()