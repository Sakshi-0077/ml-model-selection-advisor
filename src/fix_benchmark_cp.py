import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "benchmark_results.csv"
)


# ============================================================
# OFFICIAL MODELS
# ============================================================

EXPECTED_MODELS = {
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "KNN",
    "SVM",
    "XGBoost",
}


# ============================================================
# MODEL ALIASES
# ============================================================

MODEL_NAME_ALIASES = {

    "logistic_regression":
        "LogisticRegression",

    "LogisticRegression":
        "LogisticRegression",

    "decision_tree":
        "DecisionTree",

    "DecisionTree":
        "DecisionTree",

    "random_forest":
        "RandomForest",

    "RandomForest":
        "RandomForest",

    "knn":
        "KNN",

    "KNN":
        "KNN",

    "svm":
        "SVM",

    "SVM":
        "SVM",

    "xgboost":
        "XGBoost",

    "XGBoost":
        "XGBoost",
}


# ============================================================
# REQUIRED METRICS
# ============================================================

REQUIRED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "training_time",
]


# ============================================================
# NORMALIZE MODEL
# ============================================================

def normalize_model_name(model):

    if pd.isna(model):
        return None

    return MODEL_NAME_ALIASES.get(
        str(model).strip()
    )


# ============================================================
# CHECK VALID METRICS
# ============================================================

def has_valid_metrics(row):

    for metric in REQUIRED_METRICS:

        value = row.get(metric)

        if value is None or pd.isna(value):
            return False

        try:

            value = float(value)

            if not np.isfinite(value):
                return False

        except (ValueError, TypeError):

            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FIXING BENCHMARK CHECKPOINT")
    print("=" * 70)

    print()
    print(f"Checkpoint:")
    print(f"    {RESULTS_CSV}")

    if not RESULTS_CSV.exists():

        print()
        print("ERROR: benchmark_results.csv not found.")

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        RESULTS_CSV
    )

    print()
    print(
        f"Rows before cleanup: {len(df)}"
    )

    # --------------------------------------------------------
    # Normalize model names
    # --------------------------------------------------------

    df["model"] = (
        df["model"]
        .apply(normalize_model_name)
    )

    # --------------------------------------------------------
    # Remove non-official models
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["model"].isin(
            EXPECTED_MODELS
        )
    ].copy()

    print(
        f"Removed non-official rows: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Determine actual validity
    # --------------------------------------------------------

    df["metrics_valid"] = (
        df.apply(
            has_valid_metrics,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Count invalid rows
    # --------------------------------------------------------

    invalid = df[
        ~df["metrics_valid"]
    ].copy()

    valid = df[
        df["metrics_valid"]
    ].copy()

    print()
    print(
        f"Valid completed evaluations: "
        f"{len(valid)}"
    )

    print(
        f"Incomplete evaluations: "
        f"{len(invalid)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Remove incomplete rows completely.
    #
    # benchmark_all.py will recreate them.
    # --------------------------------------------------------

    if not invalid.empty:

        print()
        print(
            "Removing incomplete evaluations..."
        )

        for _, row in invalid.iterrows():

            print(
                f"    {row.get('dataset_name', 'unknown')} "
                f"| {row.get('model', 'unknown')}"
            )

    # --------------------------------------------------------
    # Keep only genuinely completed rows
    # --------------------------------------------------------

    df = valid.copy()

    df = df.drop(
        columns=["metrics_valid"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Remove duplicate combinations
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = (
        df
        .sort_values(
            by=[
                "dataset_id",
                "model"
            ]
        )
        .drop_duplicates(
            subset=[
                "dataset_id",
                "model"
            ],
            keep="last"
        )
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print()
    print(
        f"Duplicate combinations removed: "
        f"{duplicates_removed}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKPOINT CLEANUP COMPLETE")
    print("=" * 70)

    print(
        f"Rows remaining: {len(df)}"
    )

    print(
        f"Valid evaluations remaining: {len(df)}"
    )

    if not df.empty:

        print()
        print(
            "Model counts:"
        )

        print(
            df["model"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()
    print(
        "The incomplete evaluations have been removed."
    )

    print(
        "benchmark_all.py can now safely resume them."
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        f"    {RESULTS_CSV}"
    )


if __name__ == "__main__":

    main()