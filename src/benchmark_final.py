import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "benchmark_results.csv"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "benchmark_results_final.csv"
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_DATASETS = 68

EXPECTED_MODELS = 6

EXPECTED_EVALUATIONS = (
    EXPECTED_DATASETS * EXPECTED_MODELS
)


MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "KNN",
    "SVM",
    "XGBoost",
]


MODEL_DISPLAY_NAMES = {
    "LogisticRegression": "LogisticRegression",
    "DecisionTree": "DecisionTree",
    "RandomForest": "RandomForest",
    "KNN": "KNN",
    "SVM": "SVM",
    "XGBoost": "XGBoost",
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "task_id",
    "dataset_id",
    "dataset_name",
    "target_column",
    "model",
    "n_samples",
    "n_features",
    "n_classes",
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "training_time",
    "status",
    "error",
]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    print()
    print("=" * 70)
    print("LOADING EXISTING BENCHMARK RESULTS")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark results file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nInput file:"
    )

    print(
        f"    {INPUT_FILE}"
    )

    print(
        f"\nRows loaded: "
        f"{len(df)}"
    )

    print(
        f"Columns loaded: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# VALIDATE COLUMNS
# ============================================================

def validate_columns(df):

    print()
    print("=" * 70)
    print("VALIDATING COLUMNS")
    print("=" * 70)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(
                f"    - {column}"
                for column in missing_columns
            )
        )

    print(
        "✓ All required benchmark columns are present."
    )


# ============================================================
# CLEAN DATA TYPES
# ============================================================

def clean_data_types(df):

    print()
    print("=" * 70)
    print("CLEANING DATA TYPES")
    print("=" * 70)

    df = df.copy()

    integer_columns = [
        "task_id",
        "dataset_id",
        "n_samples",
        "n_features",
        "n_classes",
    ]

    float_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time",
    ]

    for column in integer_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    for column in float_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["status"] = (
        df["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(
        "✓ Data types cleaned."
    )

    return df


# ============================================================
# CLEAN MODEL NAMES
# ============================================================

def clean_model_names(df):

    print()
    print("=" * 70)
    print("CLEANING MODEL NAMES")
    print("=" * 70)

    df = df.copy()

    aliases = {
        "logistic_regression": "LogisticRegression",
        "logisticregression": "LogisticRegression",

        "decision_tree": "DecisionTree",
        "decisiontree": "DecisionTree",

        "random_forest": "RandomForest",
        "randomforest": "RandomForest",

        "knn": "KNN",

        "svm": "SVM",

        "xgboost": "XGBoost",
    }

    df["model"] = (
        df["model"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["model"] = (
        df["model"]
        .apply(
            lambda x: aliases.get(
                x.lower(),
                x
            )
            if x
            else ""
        )
    )

    before = len(df)

    df = df[
        df["model"].isin(
            MODELS
        )
    ].copy()

    removed = (
        before - len(df)
    )

    print(
        f"Rows before model filtering: "
        f"{before}"
    )

    print(
        f"Rows removed as non-final models: "
        f"{removed}"
    )

    print()
    print(
        "Models retained:"
    )

    for model in MODELS:

        count = int(
            (
                df["model"] == model
            ).sum()
        )

        print(
            f"    {model}: {count}"
        )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    print()
    print("=" * 70)
    print("CHECKING DUPLICATES")
    print("=" * 70)

    duplicate_mask = df.duplicated(
        subset=[
            "dataset_id",
            "model",
        ],
        keep="last"
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print(
        f"Duplicate dataset × model rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        print()
        print(
            "Duplicate combinations:"
        )

        duplicates = df[
            duplicate_mask
        ][
            [
                "dataset_id",
                "dataset_name",
                "model",
                "status",
            ]
        ]

        print(
            duplicates.to_string(
                index=False
            )
        )

        df = df[
            ~duplicate_mask
        ].copy()

        print()
        print(
            "✓ Duplicate rows removed."
        )

    else:

        print(
            "✓ No duplicate dataset × model combinations."
        )

    return df


# ============================================================
# NORMALIZE FAILED / INCOMPLETE RESULTS
# ============================================================

def normalize_failed_results(df):

    print()
    print("=" * 70)
    print("CHECKING FAILED DATASET RESULTS")
    print("=" * 70)

    df = df.copy()

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time",
    ]

    missing_metric_mask = (
        df[metric_columns]
        .isna()
        .any(axis=1)
    )

    successful_missing_metrics = (
        (df["status"] == "success")
        & missing_metric_mask
    )

    successful_missing_count = int(
        successful_missing_metrics.sum()
    )

    if successful_missing_count > 0:

        print()
        print(
            f"Found {successful_missing_count} successful rows "
            f"with missing benchmark metrics."
        )

        print(
            "These rows will be marked as incomplete."
        )

        df.loc[
            successful_missing_metrics,
            "status"
        ] = "incomplete"

        df.loc[
            successful_missing_metrics,
            "error"
        ] = (
            df.loc[
                successful_missing_metrics,
                "error"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        empty_error_mask = (
            successful_missing_metrics
            & (
                df["error"].fillna("").astype(str).str.strip()
                == ""
            )
        )

        df.loc[
            empty_error_mask,
            "error"
        ] = (
            "Evaluation marked successful but one or more "
            "benchmark metrics are missing."
        )

        print(
            "✓ Missing-metric success rows marked as incomplete."
        )

    failed_mask = (
        df["status"] != "success"
    )

    failed_count = int(
        failed_mask.sum()
    )

    print()
    print(
        f"Non-successful/incomplete evaluations: "
        f"{failed_count}"
    )

    if failed_count > 0:

        print()
        print(
            "Failed/incomplete dataset/model combinations:"
        )

        failed_rows = df[
            failed_mask
        ][
            [
                "dataset_id",
                "dataset_name",
                "model",
                "status",
                "error",
            ]
        ]

        print(
            failed_rows.to_string(
                index=False
            )
        )

    else:

        print(
            "✓ All evaluations are successful."
        )

    return df


# ============================================================
# CHECK DATASET × MODEL MATRIX
# ============================================================

def check_matrix(df):

    print()
    print("=" * 70)
    print("CHECKING DATASET × MODEL MATRIX")
    print("=" * 70)

    expected_models = set(
        MODELS
    )

    dataset_ids = (
        df["dataset_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    incomplete = []

    for dataset_id in sorted(
        dataset_ids
    ):

        dataset_rows = df[
            df["dataset_id"] == dataset_id
        ]

        actual_models = set(
            dataset_rows["model"]
        )

        missing = (
            expected_models
            - actual_models
        )

        if missing:

            dataset_name = (
                dataset_rows[
                    "dataset_name"
                ]
                .dropna()
            )

            if len(dataset_name) > 0:
                dataset_name = dataset_name.iloc[0]
            else:
                dataset_name = "UNKNOWN"

            incomplete.append(
                (
                    dataset_id,
                    dataset_name,
                    missing
                )
            )

    if incomplete:

        print()
        print(
            "Incomplete dataset/model combinations:"
        )

        for (
            dataset_id,
            dataset_name,
            missing
        ) in incomplete:

            print(
                f"    {dataset_id} "
                f"({dataset_name}): "
                f"missing "
                f"{', '.join(sorted(missing))}"
            )

    else:

        print(
            "✓ Every represented dataset has all 6 models."
        )

    return incomplete


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def build_final_dataset(df):

    print()
    print("=" * 70)
    print("BUILDING FINAL BENCHMARK DATASET")
    print("=" * 70)

    final_columns = [
        "task_id",
        "dataset_id",
        "dataset_name",
        "target_column",
        "model",
        "n_samples",
        "n_features",
        "n_classes",
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time",
        "status",
        "error",
        "data_source",
    ]

    available_columns = [
        column
        for column in final_columns
        if column in df.columns
    ]

    df = df[
        available_columns
    ].copy()

    model_order = {
        model: index
        for index, model in enumerate(
            MODELS
        )
    }

    df["_model_order"] = (
        df["model"]
        .map(model_order)
    )

    df = df.sort_values(
        by=[
            "dataset_id",
            "_model_order",
        ],
        na_position="last"
    )

    df = df.drop(
        columns=[
            "_model_order"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"✓ Final dataset contains "
        f"{len(df)} rows."
    )

    return df


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(df):

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    row_count = len(df)

    dataset_count = (
        df["dataset_id"]
        .nunique()
    )

    model_count = (
        df["model"]
        .nunique()
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "dataset_id",
                "model",
            ]
        ).sum()
    )

    successful_count = int(
        (
            df["status"] == "success"
        ).sum()
    )

    incomplete_count = int(
        (
            df["status"] != "success"
        ).sum()
    )

    print(
        f"Rows: "
        f"{row_count}"
    )

    print(
        f"Unique datasets: "
        f"{dataset_count}"
    )

    print(
        f"Unique models: "
        f"{model_count}"
    )

    print(
        f"Expected datasets: "
        f"{EXPECTED_DATASETS}"
    )

    print(
        f"Expected models: "
        f"{EXPECTED_MODELS}"
    )

    print(
        f"Expected evaluations: "
        f"{EXPECTED_EVALUATIONS}"
    )

    print(
        f"Duplicate combinations: "
        f"{duplicate_count}"
    )

    print(
        f"Successful evaluations: "
        f"{successful_count}"
    )

    print(
        f"Incomplete/failed evaluations: "
        f"{incomplete_count}"
    )

    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    print()
    print(
        "Evaluations per dataset:"
    )

    dataset_counts = (
        df.groupby(
            "dataset_id"
        )
        .size()
    )

    count_distribution = (
        dataset_counts
        .value_counts()
        .sort_index()
    )

    for count, datasets in (
        count_distribution.items()
    ):

        print(
            f"    {int(count)} models: "
            f"{int(datasets)} datasets"
        )

    # --------------------------------------------------------
    # Model counts
    # --------------------------------------------------------

    print()
    print(
        "Model counts:"
    )

    for model in MODELS:

        count = int(
            (
                df["model"] == model
            ).sum()
        )

        print(
            f"    {MODEL_DISPLAY_NAMES[model]}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # Status counts
    # --------------------------------------------------------

    print()
    print(
        "Status counts:"
    )

    status_counts = (
        df["status"]
        .value_counts(
            dropna=False
        )
    )

    for status, count in (
        status_counts.items()
    ):

        print(
            f"    {status}: "
            f"{int(count)}"
        )

    # --------------------------------------------------------
    # Missing metrics
    # --------------------------------------------------------

    print()
    print(
        "Missing metric values:"
    )

    for column in [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time",
    ]:

        missing = int(
            df[column]
            .isna()
            .sum()
        )

        print(
            f"    {column}: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Dataset × model matrix validation
    # --------------------------------------------------------

    dataset_model_counts = (
        df.groupby(
            "dataset_id"
        )["model"]
        .nunique()
    )

    complete_dataset_count = int(
        (
            dataset_model_counts
            == EXPECTED_MODELS
        ).sum()
    )

    incomplete_dataset_count = (
        dataset_count
        - complete_dataset_count
    )

    print()
    print(
        f"Fully completed datasets: "
        f"{complete_dataset_count}/"
        f"{EXPECTED_DATASETS}"
    )

    print(
        f"Incomplete datasets: "
        f"{incomplete_dataset_count}"
    )

    # --------------------------------------------------------
    # Check successful complete evaluations
    # --------------------------------------------------------

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time",
    ]

    successful_complete_rows = (
        (df["status"] == "success")
        & (
            ~df[metric_columns]
            .isna()
            .any(axis=1)
        )
    )

    valid_successful_count = int(
        successful_complete_rows.sum()
    )

    print()
    print(
        f"Valid successful evaluations: "
        f"{valid_successful_count}"
    )

    # --------------------------------------------------------
    # Final matrix condition
    # --------------------------------------------------------

    complete_matrix = (
        row_count
        == EXPECTED_EVALUATIONS

        and dataset_count
        == EXPECTED_DATASETS

        and model_count
        == EXPECTED_MODELS

        and duplicate_count
        == 0

        and complete_dataset_count
        == EXPECTED_DATASETS

        and successful_count
        == EXPECTED_EVALUATIONS

        and valid_successful_count
        == EXPECTED_EVALUATIONS
    )

    if complete_matrix:

        print()
        print("=" * 70)
        print("✓✓✓ 68 × 6 BENCHMARK MATRIX COMPLETE ✓✓✓")
        print("=" * 70)

        print(
            f"{EXPECTED_DATASETS} datasets × "
            f"{EXPECTED_MODELS} models = "
            f"{EXPECTED_EVALUATIONS} evaluations"
        )

    else:

        print()
        print("=" * 70)
        print("⚠ BENCHMARK MATRIX IS NOT COMPLETE")
        print("=" * 70)

        print(
            "The final file has been cleaned and validated, "
            "but some dataset/model evaluations are incomplete."
        )

        print(
            "Missing evaluations must be rerun before "
            "using the benchmark as the final 68 × 6 matrix."
        )

    return complete_matrix


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

def save_final_results(df):

    print()
    print("=" * 70)
    print("SAVING FINAL RESULTS")
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "✓ Final benchmark saved to:"
    )

    print(
        f"    {OUTPUT_FILE}"
    )

    print(
        f"\nFinal rows written: "
        f"{len(df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL 68 DATASET × 6 MODEL BENCHMARK")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script uses the existing "
        "benchmark_results.csv."
    )

    print(
        "It does NOT download datasets."
    )

    print(
        "It does NOT rerun models."
    )

    print(
        "It only cleans, filters and validates "
        "the existing benchmark results."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_results()

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    validate_columns(
        df
    )

    # --------------------------------------------------------
    # Clean data types
    # --------------------------------------------------------

    df = clean_data_types(
        df
    )

    # --------------------------------------------------------
    # Clean model names
    # --------------------------------------------------------

    df = clean_model_names(
        df
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Check failed/incomplete evaluations
    # --------------------------------------------------------

    df = normalize_failed_results(
        df
    )

    # --------------------------------------------------------
    # Check dataset/model matrix
    # --------------------------------------------------------

    check_matrix(
        df
    )

    # --------------------------------------------------------
    # Build final dataset
    # --------------------------------------------------------

    df = build_final_dataset(
        df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    complete = final_validation(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_final_results(
        df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()

    if complete:

        print("=" * 70)
        print("✓ FINAL BENCHMARK READY FOR ADVISOR")
        print("=" * 70)

    else:

        print("=" * 70)
        print("⚠ FINAL FILE SAVED, BUT MATRIX IS INCOMPLETE")
        print("=" * 70)

        print(
            "Do not assume missing evaluations are successful."
        )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()