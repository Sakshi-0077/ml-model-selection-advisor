import time
import warnings
from pathlib import Path

import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_CSV = PROCESSED_DIR / "benchmark_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

OPENML_TIMEOUT = 60
OPENML_RETRIES = 3


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


ORDERED_MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "KNN",
    "SVM",
    "XGBoost",
]


# ============================================================
# MODEL NAME ALIASES
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
# FALLBACK DATASETS
# ============================================================

FALLBACK_DATASETS = {
    "MiceProtein": 40966,
    "steel-plates-fault": 40982,
    "wilt": 40983,
    "climate-model-simulation-crashes": 40994,
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND DATASET CSV
# ============================================================

def find_dataset_csv():

    expected = RAW_DIR / "candidate_datasets.csv"

    if expected.exists():

        return expected

    candidates = list(
        DATA_DIR.rglob(
            "candidate_datasets.csv"
        )
    )

    if candidates:

        print()
        print(
            "Dataset CSV found automatically:"
        )

        print(
            f"    {candidates[0]}"
        )

        return candidates[0]

    candidates = list(
        PROJECT_ROOT.rglob(
            "candidate_datasets.csv"
        )
    )

    if candidates:

        print()
        print(
            "Dataset CSV found automatically:"
        )

        print(
            f"    {candidates[0]}"
        )

        return candidates[0]

    raise FileNotFoundError(
        "\nCould not find candidate_datasets.csv.\n\n"
        "Expected location:\n"
        f"    {expected}\n"
    )


# ============================================================
# GET MODELS
# ============================================================

def get_models():

    models = {

        "LogisticRegression": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ]),


        "DecisionTree": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                )
            )
        ]),


        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]),


        "KNN": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]),


        "SVM": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                SVC(
                    probability=True,
                    random_state=RANDOM_STATE
                )
            )
        ]),
    }


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric="logloss"
                )
            )
        ])

    return models


# ============================================================
# NORMALIZE MODEL NAME
# ============================================================

def normalize_model_name(model_name):

    if pd.isna(model_name):

        return None

    name = str(
        model_name
    ).strip()

    return MODEL_NAME_ALIASES.get(
        name
    )


# ============================================================
# OPENML METADATA
# ============================================================

def get_openml_metadata(dataset_id):

    url = (
        f"https://www.openml.org/api/v1/json/data/"
        f"{int(dataset_id)}"
    )

    last_error = None

    for attempt in range(
        1,
        OPENML_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                timeout=OPENML_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            return data["data_set_description"]

        except Exception as e:

            last_error = e

            print(
                f"        Metadata attempt "
                f"{attempt}/{OPENML_RETRIES} failed: "
                f"{type(e).__name__}"
            )

            if attempt < OPENML_RETRIES:

                time.sleep(2)

    raise RuntimeError(
        f"Could not retrieve OpenML metadata "
        f"for dataset {dataset_id}: "
        f"{last_error}"
    )


# ============================================================
# NORMAL OPENML LOADER
# ============================================================

def load_openml_dataset(dataset_id):

    from sklearn.datasets import fetch_openml

    last_error = None

    for attempt in range(
        1,
        OPENML_RETRIES + 1
    ):

        try:

            print(
                f"      Trying normal OpenML download "
                f"(attempt {attempt}/{OPENML_RETRIES})..."
            )

            data = fetch_openml(
                data_id=int(dataset_id),
                as_frame=True,
                parser="auto"
            )

            X = data.data
            y = data.target

            metadata = {
                "name": data.details.get(
                    "name",
                    str(dataset_id)
                ),

                "default_target_attribute": (
                    data.target.name
                    if hasattr(
                        data.target,
                        "name"
                    )
                    and data.target.name is not None
                    else "target"
                )
            }

            return X, y, metadata

        except Exception as e:

            last_error = e

            print(
                f"        OpenML attempt "
                f"{attempt}/{OPENML_RETRIES} failed: "
                f"{type(e).__name__}"
            )

            if attempt < OPENML_RETRIES:

                time.sleep(3)

    print(
        f"      OpenML failed completely: "
        f"{type(last_error).__name__}"
    )

    return None


# ============================================================
# PARQUET FALLBACK
# ============================================================

def load_openml_parquet(dataset_id):

    print(
        f"      Trying OpenML Parquet fallback "
        f"for dataset {dataset_id}..."
    )

    metadata = get_openml_metadata(
        dataset_id
    )

    parquet_url = metadata.get(
        "parquet_url"
    )

    if not parquet_url:

        raise RuntimeError(
            f"OpenML metadata does not provide "
            f"parquet_url for dataset {dataset_id}"
        )

    print(
        "      Parquet URL found."
    )

    df = pd.read_parquet(
        parquet_url
    )

    return df, metadata


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    dataset_id,
    dataset_name,
    target_column=None
):

    print(
        f"      Loading {dataset_name} "
        f"(OpenML ID: {dataset_id})"
    )


    # --------------------------------------------------------
    # NORMAL OPENML
    # --------------------------------------------------------

    result = load_openml_dataset(
        dataset_id
    )

    if result is not None:

        X, y, metadata = result

        print(
            "      ✓ Loaded using normal OpenML"
        )

        return (
            X,
            y,
            metadata,
            "openml"
        )


    # --------------------------------------------------------
    # PARQUET FALLBACK
    # --------------------------------------------------------

    if dataset_name in FALLBACK_DATASETS:

        try:

            df, metadata = load_openml_parquet(
                dataset_id
            )

            target_name = (
                metadata.get(
                    "default_target_attribute"
                )
                or target_column
            )

            if target_name is None:

                raise RuntimeError(
                    "Could not determine target column."
                )

            if target_name not in df.columns:

                raise RuntimeError(
                    f"Target '{target_name}' "
                    f"not found in dataframe."
                )

            X = df.drop(
                columns=[target_name]
            )

            y = df[target_name]

            row_id = metadata.get(
                "row_id_attribute"
            )

            if (
                row_id
                and row_id in X.columns
            ):

                X = X.drop(
                    columns=[row_id]
                )

            ignored = metadata.get(
                "ignore_attribute",
                []
            )

            if isinstance(
                ignored,
                str
            ):

                ignored = [
                    x.strip()
                    for x in ignored.split(",")
                ]

            ignored = [
                col
                for col in ignored
                if col in X.columns
            ]

            if ignored:

                X = X.drop(
                    columns=ignored
                )

            print(
                "      ✓ Loaded using OpenML "
                "Parquet fallback"
            )

            return (
                X,
                y,
                metadata,
                "parquet"
            )

        except Exception as e:

            print(
                "      ✗ Parquet fallback failed:"
            )

            print(
                f"        {type(e).__name__}: {e}"
            )


    raise RuntimeError(
        f"Could not load dataset "
        f"{dataset_name} ({dataset_id})"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(X, y):

    X = X.copy()
    y = y.copy()


    # --------------------------------------------------------
    # REMOVE COMPLETELY EMPTY COLUMNS
    # --------------------------------------------------------

    empty_columns = [
        col
        for col in X.columns
        if X[col].isna().all()
    ]

    if empty_columns:

        X = X.drop(
            columns=empty_columns
        )

        print(
            f"      Removed "
            f"{len(empty_columns)} completely "
            f"empty feature(s)."
        )


    # --------------------------------------------------------
    # REMOVE CONSTANT COLUMNS
    # --------------------------------------------------------

    constant_columns = [
        col
        for col in X.columns
        if X[col].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:

        X = X.drop(
            columns=constant_columns
        )

        print(
            f"      Removed "
            f"{len(constant_columns)} constant feature(s)."
        )


    # --------------------------------------------------------
    # CLEAN TARGET FIRST
    # --------------------------------------------------------

    y = y.astype(str)

    valid_target = (
        y.notna()
        & (y != "nan")
        & (y != "None")
        & (y != "")
    )

    X = X.loc[
        valid_target
    ].reset_index(
        drop=True
    )

    y = y.loc[
        valid_target
    ].reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # CONVERT FEATURES
    # --------------------------------------------------------

    for col in X.columns:

        series = X[col]


        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            X[col] = (
                series.astype("int64")
                .replace(
                    -9223372036854775808,
                    np.nan
                )
            )

            continue


        # ----------------------------------------------------
        # Boolean
        # ----------------------------------------------------

        if pd.api.types.is_bool_dtype(
            series
        ):

            X[col] = series.astype(float)

            continue


        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            X[col] = pd.to_numeric(
                series,
                errors="coerce"
            )

            continue


        # ----------------------------------------------------
        # Try numeric conversion
        # ----------------------------------------------------

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        numeric_ratio = (
            numeric_series.notna().mean()
        )

        if numeric_ratio >= 0.95:

            X[col] = numeric_series

            continue


        # ----------------------------------------------------
        # Categorical encoding
        # ----------------------------------------------------

        X[col] = (
            series
            .astype("category")
            .cat.codes
            .replace(
                -1,
                np.nan
            )
            .astype(float)
        )


    # --------------------------------------------------------
    # REPLACE INFINITY
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # --------------------------------------------------------
    # REMOVE COLUMNS THAT BECAME COMPLETELY EMPTY
    # --------------------------------------------------------

    empty_after_conversion = [
        col
        for col in X.columns
        if X[col].isna().all()
    ]

    if empty_after_conversion:

        X = X.drop(
            columns=empty_after_conversion
        )

        print(
            f"      Removed "
            f"{len(empty_after_conversion)} "
            f"non-convertible feature(s)."
        )


    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    return X, y_encoded


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    start_time = time.time()

    try:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )


        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        elapsed = (
            time.time()
            - start_time
        )


        print(
            f"        ✓ {model_name} "
            f"| accuracy={accuracy:.4f} "
            f"| balanced_accuracy="
            f"{balanced_accuracy:.4f} "
            f"| F1={f1:.4f} "
            f"| time={elapsed:.2f}s"
        )


        return {

            "model": model_name,

            "accuracy": accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "f1_score": f1,

            "precision": precision,

            "recall": recall,

            "training_time": elapsed,

            "status": "success",

            "error": ""
        }


    except Exception as e:

        elapsed = (
            time.time()
            - start_time
        )

        error_message = (
            f"{type(e).__name__}: "
            f"{str(e)[:500]}"
        )


        print(
            f"        ✗ {model_name}: "
            f"{error_message}"
        )


        return {

            "model": model_name,

            "accuracy": np.nan,

            "balanced_accuracy": np.nan,

            "f1_score": np.nan,

            "precision": np.nan,

            "recall": np.nan,

            "training_time": elapsed,

            "status": "failed",

            "error": error_message
        }


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if not RESULTS_CSV.exists():

        print()
        print(
            "No previous benchmark results found."
        )

        print(
            "Starting benchmark."
        )

        return []


    try:

        df = pd.read_csv(
            RESULTS_CSV
        )

        if df.empty:

            return []


        print()
        print(
            f"Existing checkpoint found: "
            f"{len(df)} rows."
        )


        if "status" in df.columns:

            success_count = int(
                (
                    df["status"]
                    .astype(str)
                    .str.lower()
                    == "success"
                ).sum()
            )

            print(
                f"Successful evaluations: "
                f"{success_count}"
            )


        # ----------------------------------------------------
        # Normalize model names
        # ----------------------------------------------------

        if "model" in df.columns:

            df["model"] = (
                df["model"]
                .apply(
                    normalize_model_name
                )
            )


        # ----------------------------------------------------
        # Remove rows that are not official
        # ----------------------------------------------------

        df = df[
            df["model"].isin(
                EXPECTED_MODELS
            )
        ].copy()


        # ----------------------------------------------------
        # Keep ONLY latest row per
        # dataset + model
        # ----------------------------------------------------

        df["_row_order"] = np.arange(
            len(df)
        )

        df = (
            df
            .sort_values(
                "_row_order"
            )
            .drop_duplicates(
                subset=[
                    "dataset_id",
                    "model"
                ],
                keep="last"
            )
            .drop(
                columns=["_row_order"]
            )
        )


        print(
            f"Checkpoint after cleanup: "
            f"{len(df)} rows."
        )


        return df.to_dict(
            orient="records"
        )


    except Exception as e:

        print()
        print(
            "WARNING: Could not read "
            f"existing results: {e}"
        )

        return []


# ============================================================
# FIND EXISTING RESULT
# ============================================================

def find_existing_result(
    results,
    dataset_id,
    model_name
):

    target_id = int(
        dataset_id
    )

    normalized_model = (
        normalize_model_name(
            model_name
        )
    )

    for row in results:

        try:

            row_dataset_id = int(
                float(
                    row.get(
                        "dataset_id"
                    )
                )
            )

        except Exception:

            continue


        row_model = normalize_model_name(
            row.get(
                "model"
            )
        )


        if (
            row_dataset_id == target_id
            and row_model == normalized_model
        ):

            return row


    return None

# ============================================================
# GET COMPLETED MODELS FOR DATASET
# ============================================================

def get_completed_models(
    existing_results,
    dataset_id,
    dataset_name=None
):

    completed_models = set()

    try:
        target_id = int(float(dataset_id))
    except (ValueError, TypeError):
        target_id = None

    target_name = str(
        dataset_name
        if dataset_name is not None
        else ""
    ).strip().lower()

    required_metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time"
    ]

    for row in existing_results:

        # ----------------------------------------------------
        # Dataset ID
        # ----------------------------------------------------

        try:
            existing_id = int(
                float(
                    row.get("dataset_id")
                )
            )
        except (ValueError, TypeError):
            existing_id = None

        id_match = (
            target_id is not None
            and existing_id is not None
            and target_id == existing_id
        )

        # ----------------------------------------------------
        # Dataset name
        # ----------------------------------------------------

        existing_name = str(
            row.get(
                "dataset_name",
                row.get("dataset", "")
            )
        ).strip().lower()

        name_match = (
            bool(target_name)
            and existing_name == target_name
        )

        # ----------------------------------------------------
        # Dataset must match
        # ----------------------------------------------------

        if not (id_match or name_match):
            continue

        # ----------------------------------------------------
        # Normalize model
        # ----------------------------------------------------

        normalized_model = normalize_model_name(
            row.get("model")
        )

        if normalized_model not in EXPECTED_MODELS:
            continue

        # ----------------------------------------------------
        # SUCCESS MUST ALSO HAVE VALID METRICS
        # ----------------------------------------------------

        status = str(
            row.get("status", "")
        ).strip().lower()

        if status != "success":
            continue

        metrics_valid = True

        for metric in required_metrics:

            value = row.get(metric)

            if (
                value is None
                or pd.isna(value)
            ):
                metrics_valid = False
                break

            try:
                value = float(value)

                if not np.isfinite(value):
                    metrics_valid = False
                    break

            except (ValueError, TypeError):
                metrics_valid = False
                break

        # ----------------------------------------------------
        # Only genuinely completed evaluation counts
        # ----------------------------------------------------

        if metrics_valid:
            completed_models.add(
                normalized_model
            )

    return completed_models


# ============================================================
# GET MISSING MODELS
# ============================================================

def get_missing_models(
    existing_results,
    dataset_id,
    dataset_name=None
):

    completed_models = (
        get_completed_models(
            existing_results,
            dataset_id,
            dataset_name
        )
    )

    return (
        EXPECTED_MODELS
        - completed_models
    )


# ============================================================
# CHECKPOINT SAVE
#
# IMPORTANT:
# Replace old result for the same
# dataset + model instead of appending
# another duplicate.
# ============================================================

def save_result_checkpoint(
    all_results,
    result
):

    dataset_id = int(
        result["dataset_id"]
    )

    model_name = normalize_model_name(
        result["model"]
    )

    result["model"] = model_name


    replaced = False

    for index, existing in enumerate(
        all_results
    ):

        try:

            existing_id = int(
                float(
                    existing.get(
                        "dataset_id"
                    )
                )
            )

        except Exception:

            continue


        existing_model = (
            normalize_model_name(
                existing.get(
                    "model"
                )
            )
        )


        if (
            existing_id == dataset_id
            and existing_model == model_name
        ):

            all_results[index] = result

            replaced = True

            break


    if not replaced:

        all_results.append(
            result
        )


    # --------------------------------------------------------
    # Normalize ALL results
    # --------------------------------------------------------

    cleaned = []

    seen = set()


    for row in all_results:

        try:

            row_id = int(
                float(
                    row.get(
                        "dataset_id"
                    )
                )
            )

        except Exception:

            continue


        row_model = normalize_model_name(
            row.get(
                "model"
            )
        )


        if row_model not in EXPECTED_MODELS:

            continue


        key = (
            row_id,
            row_model
        )


        if key in seen:

            continue


        seen.add(key)

        row["dataset_id"] = row_id
        row["model"] = row_model

        cleaned.append(
            row
        )


    all_results.clear()

    all_results.extend(
        cleaned
    )


    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    pd.DataFrame(
        all_results
    ).to_csv(
        RESULTS_CSV,
        index=False
    )


# ============================================================
# BENCHMARK DATASET
# ============================================================

def benchmark_dataset(
    task_id,
    dataset_id,
    dataset_name,
    target_column,
    missing_models,
    all_results
):

    print()
    print("=" * 70)

    print(
        f"Benchmarking {dataset_name} "
        f"(task {task_id}, dataset_id {dataset_id})"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        (
            X,
            y,
            metadata,
            source
        ) = load_dataset(
            dataset_id,
            dataset_name,
            target_column
        )

    except Exception as e:

        print()
        print(
            f"      DATASET FAILED: "
            f"{type(e).__name__}: {e}"
        )

        return False


    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    try:

        X, y = prepare_features(
            X,
            y
        )

    except Exception as e:

        print()
        print(
            f"      PREPROCESSING FAILED: "
            f"{type(e).__name__}: {e}"
        )

        return False


    n_samples = X.shape[0]

    n_features = X.shape[1]

    n_classes = len(
        np.unique(y)
    )


    print(
        f"      Samples: {n_samples}"
    )

    print(
        f"      Features: {n_features}"
    )

    print(
        f"      Classes: {n_classes}"
    )

    print(
        f"      Source: {source}"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if n_samples < 10:

        print(
            "      DATASET FAILED: "
            "Too few samples."
        )

        return False


    if n_features == 0:

        print(
            "      DATASET FAILED: "
            "No usable features."
        )

        return False


    if n_classes < 2:

        print(
            "      DATASET FAILED: "
            "Target contains fewer than 2 classes."
        )

        return False


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

    except ValueError as e:

        print(
            "      Stratified split failed."
        )

        print(
            f"      Reason: {e}"
        )

        print(
            "      Trying normal random split..."
        )

        try:

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

        except Exception as split_error:

            print(
                f"      SPLIT FAILED: "
                f"{type(split_error).__name__}: "
                f"{split_error}"
            )

            return False


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = get_models()


    # --------------------------------------------------------
    # XGBOOST CHECK
    # --------------------------------------------------------

    if (
        "XGBoost" in missing_models
        and not XGBOOST_AVAILABLE
    ):

        print()
        print(
            "ERROR: XGBoost is not installed."
        )

        print(
            "Install it with:"
        )

        print(
            "    pip install xgboost"
        )

        missing_models = (
            missing_models
            - {"XGBoost"}
        )


    # --------------------------------------------------------
    # RUN ONLY MISSING MODELS
    # --------------------------------------------------------

    models_run = 0


    for model_name in ORDERED_MODELS:

        if model_name not in missing_models:

            continue


        model = models.get(
            model_name
        )


        if model is None:

            continue


        print()
        print(
            f"      → {model_name}"
        )


        result = benchmark_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )


        # ----------------------------------------------------
        # DATASET METADATA
        # ----------------------------------------------------

        result.update({

            "task_id":
                task_id,

            "dataset_id":
                dataset_id,

            "dataset_name":
                dataset_name,

            "target_column":
                metadata.get(
                    "default_target_attribute",
                    target_column
                ),

            "n_samples":
                n_samples,

            "n_features":
                n_features,

            "n_classes":
                n_classes,

            "data_source":
                source
        })


        # ----------------------------------------------------
        # IMMEDIATE CHECKPOINT
        # ----------------------------------------------------

        save_result_checkpoint(
            all_results,
            result
        )


        print(
            "        ✓ Checkpoint saved."
        )


        models_run += 1


    return models_run > 0


# ============================================================
# FINAL VALIDATION
# ============================================================

def print_final_validation(
    all_results,
    datasets
):

    df = pd.DataFrame(
        all_results
    )


    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)


    if df.empty:

        print(
            "No benchmark results available."
        )

        return


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    df["official_model"] = (
        df["model"]
        .apply(
            normalize_model_name
        )
    )


    df = df[
        df["official_model"]
        .isin(
            EXPECTED_MODELS
        )
    ].copy()


    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "dataset_id",
                "official_model"
            ],
            keep="last"
        )
    )


    expected_evaluations = (
        len(datasets)
        * len(EXPECTED_MODELS)
    )


    print(
        f"Rows: {len(df)}"
    )


    print(
        f"Unique datasets: "
        f"{df['dataset_id'].nunique()}"
    )


    print(
        f"Unique models: "
        f"{df['official_model'].nunique()}"
    )


    print(
        f"Expected datasets: "
        f"{len(datasets)}"
    )


    print(
        f"Expected models: "
        f"{len(EXPECTED_MODELS)}"
    )


    print(
        f"Expected evaluations: "
        f"{expected_evaluations}"
    )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            subset=[
                "dataset_id",
                "official_model"
            ]
        ).sum()
    )


    print(
        f"Duplicate combinations: "
        f"{duplicate_count}"
    )


    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    successful = df[
        df["status"]
        .astype(str)
        .str.lower()
        == "success"
    ].copy()


    print()
    print(
        f"Successful evaluations: "
        f"{len(successful)}"
    )


    # --------------------------------------------------------
    # FAILED / INCOMPLETE
    # --------------------------------------------------------

    incomplete = df[
        ~(
            df["status"]
            .astype(str)
            .str.lower()
            == "success"
        )
    ].copy()


    print(
        f"Incomplete/failed evaluations: "
        f"{len(incomplete)}"
    )


    # --------------------------------------------------------
    # MODEL COUNTS
    # --------------------------------------------------------

    print()
    print(
        "Model counts:"
    )


    model_counts = (
        df.groupby(
            "official_model"
        ).size()
    )


    for model_name in ORDERED_MODELS:

        count = int(
            model_counts.get(
                model_name,
                0
            )
        )

        print(
            f"    {model_name}: "
            f"{count}"
        )


    # --------------------------------------------------------
    # SUCCESSFUL MODEL COUNTS
    # --------------------------------------------------------

    print()
    print(
        "Successful model counts:"
    )


    success_model_counts = (
        successful
        .groupby(
            "official_model"
        )
        .size()
    )


    for model_name in ORDERED_MODELS:

        count = int(
            success_model_counts.get(
                model_name,
                0
            )
        )

        print(
            f"    {model_name}: "
            f"{count}/{len(datasets)}"
        )


    # --------------------------------------------------------
    # STATUS COUNTS
    # --------------------------------------------------------

    print()
    print(
        "Status counts:"
    )


    status_counts = (
        df["status"]
        .astype(str)
        .str.lower()
        .value_counts()
    )


    for status, count in status_counts.items():

        print(
            f"    {status}: {count}"
        )


    # --------------------------------------------------------
    # MISSING METRICS
    # --------------------------------------------------------

    print()
    print(
        "Missing metric values:"
    )


    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "training_time"
    ]


    for metric in metric_columns:

        missing = int(
            df[metric]
            .isna()
            .sum()
        )

        print(
            f"    {metric}: {missing}"
        )


    # --------------------------------------------------------
    # DATASET COMPLETION
    #
    # IMPORTANT:
    # A dataset is COMPLETE only if
    # all six models SUCCESSFULLY ran.
    # --------------------------------------------------------

    successful_counts = (
        successful
        .groupby(
            "dataset_id"
        )["official_model"]
        .nunique()
    )


    completed_datasets = int(
        (
            successful_counts
            == len(EXPECTED_MODELS)
        ).sum()
    )


    incomplete_datasets = (
        len(datasets)
        - completed_datasets
    )


    print()
    print(
        f"Fully completed datasets: "
        f"{completed_datasets}/{len(datasets)}"
    )


    print(
        f"Incomplete datasets: "
        f"{incomplete_datasets}"
    )


    # --------------------------------------------------------
    # PRINT INCOMPLETE DATASETS
    # --------------------------------------------------------

    if incomplete_datasets > 0:

        print()
        print(
            "Incomplete datasets:"
        )


        for _, row in datasets.iterrows():

            dataset_id = int(
                row["dataset_id"]
            )


            success_count = int(
                successful_counts.get(
                    dataset_id,
                    0
                )
            )


            if (
                success_count
                < len(EXPECTED_MODELS)
            ):

                missing = (
                    EXPECTED_MODELS
                    -
                    set(
                        successful[
                            successful[
                                "dataset_id"
                            ]
                            == dataset_id
                        ]["official_model"]
                    )
                )


                print(
                    f"    {row['dataset_name']}: "
                    f"{success_count}/6"
                )


                if missing:

                    print(
                        "        Missing: "
                        + ", ".join(
                            sorted(
                                missing
                            )
                        )
                    )


    # --------------------------------------------------------
    # BENCHMARK MATRIX STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)


    if len(successful) == expected_evaluations:

        print(
            "✓ BENCHMARK MATRIX IS COMPLETE"
        )

        print(
            f"✓ {len(datasets)} datasets × "
            f"{len(EXPECTED_MODELS)} models "
            f"= {expected_evaluations} successful evaluations"
        )

    else:

        missing_evaluations = (
            expected_evaluations
            - len(successful)
        )


        print(
            "⚠ BENCHMARK MATRIX IS NOT COMPLETE"
        )


        print(
            f"Successful: "
            f"{len(successful)}/{expected_evaluations}"
        )


        print(
            f"Still missing: "
            f"{missing_evaluations}"
        )


        print()
        print(
            "Run the benchmark again to retry "
            "the failed evaluations."
        )


    print("=" * 70)


    # --------------------------------------------------------
    # AVERAGE PERFORMANCE
    # --------------------------------------------------------

    if not successful.empty:

        print()
        print(
            "Average benchmark performance:"
        )


        ranking = (
            successful
            .groupby(
                "official_model"
            )
            .agg(

                mean_accuracy=(
                    "accuracy",
                    "mean"
                ),

                mean_balanced_accuracy=(
                    "balanced_accuracy",
                    "mean"
                ),

                mean_f1=(
                    "f1_score",
                    "mean"
                ),

                mean_precision=(
                    "precision",
                    "mean"
                ),

                mean_recall=(
                    "recall",
                    "mean"
                ),

                mean_training_time=(
                    "training_time",
                    "mean"
                )
            )
            .sort_values(
                "mean_accuracy",
                ascending=False
            )
        )


        print(
            ranking.to_string(
                float_format=lambda x:
                f"{x:.4f}"
            )
        )


        print()
        print(
            "Current ranking by mean accuracy:"
        )


        for position, model_name in enumerate(
            ranking.index,
            start=1
        ):

            print(
                f"    {position}. "
                f"{model_name} "
                f"("
                f"{ranking.loc[model_name, 'mean_accuracy']:.4f}"
                f")"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ML MODEL SELECTION ADVISOR"
    )
    print(
        "68-DATASET / 6-MODEL BENCHMARK"
    )
    print("=" * 70)


    create_directories()


    # --------------------------------------------------------
    # XGBOOST CHECK
    # --------------------------------------------------------

    if not XGBOOST_AVAILABLE:

        print()
        print(
            "WARNING: XGBoost is not installed."
        )

        print(
            "Install it inside your virtual environment:"
        )

        print(
            "    pip install xgboost"
        )

        print()


    # --------------------------------------------------------
    # DATASET CSV
    # --------------------------------------------------------

    try:

        dataset_csv = (
            find_dataset_csv()
        )

    except Exception as e:

        print()
        print(
            f"DATASET CSV ERROR: {e}"
        )

        return


    print()
    print(
        "Dataset list:"
    )

    print(
        f"    {dataset_csv}"
    )


    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    try:

        datasets = pd.read_csv(
            dataset_csv
        )

    except Exception as e:

        print()
        print(
            f"Could not read dataset CSV: "
            f"{type(e).__name__}: {e}"
        )

        return


    print()
    print(
        f"Loaded {len(datasets)} datasets."
    )


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "task_id",
        "dataset_id",
        "dataset_name"
    ]


    missing_columns = [
        col
        for col in required_columns
        if col not in datasets.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    all_results = (
        load_existing_results()
    )


    total = len(
        datasets
    )


    newly_benchmarked = 0
    skipped = 0
    resumed = 0
    failed = 0


    # ========================================================
    # DATASET LOOP
    # ========================================================

    for index, row in datasets.iterrows():

        dataset_number = (
            index + 1
        )


        try:

            task_id = int(
                row["task_id"]
            )


            dataset_id = int(
                row["dataset_id"]
            )


            dataset_name = str(
                row["dataset_name"]
            ).strip()


            target_column = None


            if (
                "target_column"
                in datasets.columns
            ):

                if pd.notna(
                    row["target_column"]
                ):

                    target_column = str(
                        row["target_column"]
                    )


            # ------------------------------------------------
            # SUCCESSFUL MODELS
            # ------------------------------------------------

            completed_models = (
                get_completed_models(
                    all_results,
                    dataset_id,
                    dataset_name
                )
            )


            # ------------------------------------------------
            # MISSING MODELS
            # ------------------------------------------------

            missing_models = (
                EXPECTED_MODELS
                - completed_models
            )


            # ------------------------------------------------
            # COMPLETE
            # ------------------------------------------------

            if not missing_models:

                print()
                print(
                    f"[{dataset_number}/{total}] "
                    f"SKIPPING {dataset_name} "
                    f"(6/6 SUCCESSFUL)"
                )

                skipped += 1

                continue


            # ------------------------------------------------
            # RESUME
            # ------------------------------------------------

            if completed_models:

                resumed += 1

                print()
                print(
                    f"[{dataset_number}/{total}] "
                    f"RESUMING {dataset_name}"
                )

                print(
                    f"      Successful: "
                    f"{len(completed_models)}/6"
                )

                print(
                    f"      To run: "
                    f"{len(missing_models)}/6"
                )

                print(
                    "      Missing models: "
                    + ", ".join(
                        sorted(
                            missing_models
                        )
                    )
                )


            # ------------------------------------------------
            # NEW DATASET
            # ------------------------------------------------

            else:

                print()
                print(
                    f"[{dataset_number}/{total}] "
                    f"STARTING {dataset_name}"
                )

                print(
                    "      All 6 models required."
                )


            # ------------------------------------------------
            # RUN
            # ------------------------------------------------

            result_generated = (
                benchmark_dataset(
                    task_id=task_id,

                    dataset_id=dataset_id,

                    dataset_name=dataset_name,

                    target_column=target_column,

                    missing_models=missing_models,

                    all_results=all_results
                )
            )


            if result_generated:

                newly_benchmarked += 1

                print()
                print(
                    f"      ✓ Dataset processing complete: "
                    f"{dataset_name}"
                )


            else:

                failed += 1

                print()
                print(
                    f"      ✗ No new model results generated "
                    f"for {dataset_name}"
                )


        except Exception as e:

            failed += 1

            print()
            print(
                f"      ✗ UNEXPECTED ERROR "
                f"for dataset "
                f"{row.get('dataset_name', 'unknown')}:"
            )

            print(
                f"        {type(e).__name__}: {e}"
            )

            print(
                "      Continuing to next dataset..."
            )

            continue


    # ========================================================
    # FINAL SAVE
    # ========================================================

    if all_results:

        pd.DataFrame(
            all_results
        ).to_csv(
            RESULTS_CSV,
            index=False
        )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print_final_validation(
        all_results,
        datasets
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BENCHMARK RUN FINISHED"
    )
    print("=" * 70)


    print(
        f"Datasets in input: "
        f"{total}"
    )


    print(
        f"Datasets requiring work: "
        f"{newly_benchmarked}"
    )


    print(
        f"Datasets skipped: "
        f"{skipped}"
    )


    print(
        f"Datasets partially resumed: "
        f"{resumed}"
    )


    print(
        f"Dataset-level failures: "
        f"{failed}"
    )


    print()
    print(
        "Checkpoint file:"
    )


    print(
        f"    {RESULTS_CSV}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()