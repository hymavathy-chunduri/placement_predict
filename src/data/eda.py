import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ingest import load_and_validate_data


def perform_eda(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset dimensions
    print("\n1. DATASET DIMENSIONS")
    print("-" * 40)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Data types
    print("\n2. DATA TYPES")
    print("-" * 40)
    print(df.dtypes)

    # 3. Missing values
    print("\n3. MISSING VALUES")
    print("-" * 40)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    if missing_values.sum() == 0:
        print("No missing values found.")

    # 4. Duplicate records
    print("\n4. DUPLICATE RECORDS")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # 5. Statistical summary
    print("\n5. STATISTICAL SUMMARY")
    print("-" * 40)
    print(df.describe())

    # 6. Placement class distribution
    print("\n6. PLACEMENT CLASS DISTRIBUTION")
    print("-" * 40)
    print(df["placement_status"].value_counts())
    print("\nPercentage distribution:")
    print(df["placement_status"].value_counts(normalize=True) * 100)

    # Create reports directory
    reports_dir = os.path.join("reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)

    # 7. Correlation heatmap
    print("\n7. GENERATING CORRELATION HEATMAP")

    numeric_df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(reports_dir, "correlation_heatmap.png")
    )
    plt.close()

    # 8. CGPA vs Salary scatter plot
    print("8. GENERATING CGPA VS SALARY SCATTER PLOT")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="cgpa",
        y="salary_package_lpa",
        hue="placement_status"
    )

    plt.title("CGPA vs Salary Package")
    plt.xlabel("CGPA")
    plt.ylabel("Salary Package (LPA)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(reports_dir, "scatter_cgpa_salary.png")
    )
    plt.close()

    # 9. Pair plot
    print("9. GENERATING PAIR PLOT")

    pair_columns = [
        "cgpa",
        "coding_skill_score",
        "communication_skill_score",
        "internships_count",
        "placement_status"
    ]

    pair_df = df[pair_columns].copy()

    pair_df["placement_status"] = pair_df["placement_status"].astype(str)

    sns.pairplot(
        pair_df,
        hue="placement_status"
    )

    plt.savefig(
        os.path.join(reports_dir, "pairplot_features.png")
    )
    plt.close()

    # 10. Outlier boxplots
    print("10. GENERATING OUTLIER BOXPLOTS")

    outlier_columns = [
        "cgpa",
        "coding_skill_score",
        "communication_skill_score",
        "internships_count",
        "projects_count",
        "salary_package_lpa"
    ]

    plt.figure(figsize=(14, 8))

    df[outlier_columns].boxplot()

    plt.title("Outlier Detection using Boxplots")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(reports_dir, "outliers_boxplot.png")
    )
    plt.close()

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nReports saved in: {reports_dir}")


if __name__ == "__main__":

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    try:
        raw_data = load_and_validate_data(DATA_PATH)
        perform_eda(raw_data)

    except Exception as e:
        print(f"\nEDA lifecycle termination: {str(e)}")