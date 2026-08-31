from pathlib import Path
import pandas as pd
import joblib
import numpy as np

from sklearn.utils.multiclass import type_of_target


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_SELECTOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_selector.pkl"
)


# =========================================================
# TARGET TYPE DETECTION
# =========================================================

def detect_task_type(y):

    y = y.dropna()

    if y.empty:
        raise ValueError(
            "Target column contains no valid values."
        )

    target_type = type_of_target(y)

    # Classification
    if target_type in [
        "binary",
        "multiclass",
        "multiclass-multioutput"
    ]:
        return "classification"

    # Regression
    if target_type in [
        "continuous",
        "continuous-multioutput"
    ]:
        return "regression"

    # Fallback for object/category/string targets
    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or y.dtype == "bool"
    ):
        return "classification"

    raise ValueError(
        f"Unsupported target type: {target_type}"
    )


# =========================================================
# LOAD CLASSIFICATION SELECTOR
# =========================================================

def load_selector():

    if not MODEL_SELECTOR_PATH.exists():
        return None, None

    try:

        selector_data = joblib.load(
            MODEL_SELECTOR_PATH
        )

        selector = selector_data["model"]
        features = selector_data["features"]

        return selector, features

    except Exception:

        return None, None


# =========================================================
# CLASSIFICATION HEURISTIC
# =========================================================

def classification_heuristic(
    n_samples,
    n_features,
    n_classes,
    numeric_features,
    categorical_features
):

    # Small datasets
    if n_samples < 500:

        if numeric_features > categorical_features:
            return "SVM"

        return "Logistic Regression"

    # Large categorical datasets
    if categorical_features > numeric_features:
        return "Random Forest"

    # High dimensional data
    if n_features > 30:
        return "Random Forest"

    # Moderate numeric datasets
    if n_samples >= 500:
        return "Random Forest"

    return "SVM"


# =========================================================
# REGRESSION HEURISTIC
# =========================================================

def regression_heuristic(
    n_samples,
    n_features,
    numeric_features,
    categorical_features
):

    # Mostly categorical
    if categorical_features > numeric_features:

        if n_samples >= 500:
            return "Random Forest"

        return "Gradient Boosting"

    # Small numeric dataset
    if n_samples < 500:

        if n_features <= 20:
            return "Random Forest"

        return "Gradient Boosting"

    # Large numeric dataset
    if n_samples >= 5000:

        return "Random Forest"

    return "Gradient Boosting"


# =========================================================
# RECOMMEND MODEL
# =========================================================

def recommend_model(
    dataset_path,
    target_column,
    return_details=False
):

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n{dataset_path}"
        )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    try:

        df = pd.read_csv(
            dataset_path
        )

    except Exception as e:

        raise ValueError(
            f"\nCould not read CSV file.\n"
            f"Reason: {e}"
        )

    if df.empty:

        raise ValueError(
            "\nThe CSV file is empty."
        )

    # -----------------------------------------------------
    # TARGET VALIDATION
    # -----------------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"\nTarget column '{target_column}' "
            "not found.\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # -----------------------------------------------------
    # REMOVE MISSING TARGET VALUES
    # -----------------------------------------------------

    df = df.dropna(
        subset=[target_column]
    )

    if df.empty:

        raise ValueError(
            "No rows remain after removing "
            "missing target values."
        )

    # -----------------------------------------------------
    # FEATURES / TARGET
    # -----------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # -----------------------------------------------------
    # TASK TYPE
    # -----------------------------------------------------

    task_type = detect_task_type(
        y
    )

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    n_samples = len(X)

    n_features = X.shape[1]

    n_classes = (
        y.nunique()
        if task_type == "classification"
        else 0
    )

    numeric_features = X.select_dtypes(
        include=np.number
    ).shape[1]

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).shape[1]

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    total_cells = (
        X.shape[0]
        * X.shape[1]
    )

    if total_cells > 0:

        missing_percentage = (
            X.isnull().sum().sum()
            / total_cells
        ) * 100

    else:

        missing_percentage = 0.0

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if task_type == "classification":

        selector, features = load_selector()

        recommended_model = None
        ranking = None

        # -----------------------------------------------
        # Try trained selector
        # -----------------------------------------------

        if selector is not None:

            try:

                selector_input = pd.DataFrame([
                    {
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_classes": n_classes,
                        "numeric_features": numeric_features,
                        "categorical_features": categorical_features,
                        "missing_percentage": missing_percentage,
                    }
                ])

                selector_input = selector_input[
                    features
                ]

                recommended_model = selector.predict(
                    selector_input
                )[0]

                if hasattr(
                    selector,
                    "predict_proba"
                ):

                    probabilities = (
                        selector.predict_proba(
                            selector_input
                        )[0]
                    )

                    classes = selector.classes_

                    ranking = pd.Series(
                        probabilities,
                        index=classes
                    ).sort_values(
                        ascending=False
                    )

            except Exception:

                recommended_model = None

        # -----------------------------------------------
        # Fallback
        # -----------------------------------------------

        if recommended_model is None:

            recommended_model = classification_heuristic(
                n_samples,
                n_features,
                n_classes,
                numeric_features,
                categorical_features
            )

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        recommended_model = regression_heuristic(
            n_samples,
            n_features,
            numeric_features,
            categorical_features
        )

        ranking = None

    # =====================================================
    # DETAILS
    # =====================================================

    details = {

        "task_type": task_type,

        "dataset": dataset_path.name,

        "samples": n_samples,

        "features": n_features,

        "classes": n_classes,

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "missing_percentage": missing_percentage,

        "recommended_model": str(
            recommended_model
        ),

        "ranking": ranking,

    }

    # =====================================================
    # TERMINAL OUTPUT
    # =====================================================

    print("\n" + "=" * 70)
    print("MODEL SELECTION ADVISOR")
    print("=" * 70)

    print(
        f"\nDataset: {dataset_path.name}"
    )

    print(
        f"Samples: {n_samples}"
    )

    print(
        f"Features: {n_features}"
    )

    print(
        f"Task type: {task_type}"
    )

    if task_type == "classification":

        print(
            f"Classes: {n_classes}"
        )

    print(
        f"Numeric features: {numeric_features}"
    )

    print(
        f"Categorical features: {categorical_features}"
    )

    print(
        f"Missing values: "
        f"{missing_percentage:.2f}%"
    )

    print(
        "\nRecommended model:"
    )

    print(
        recommended_model
    )

    if ranking is not None:

        print(
            "\nModel ranking:"
        )

        for model, probability in ranking.items():

            print(
                f"{model}: "
                f"{probability * 100:.2f}%"
            )

    print("=" * 70)

    if return_details:

        return details

    return str(
        recommended_model
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("MODEL SELECTION ADVISOR")
    print("=" * 70)

    dataset_path = input(
        "\nEnter CSV file path: "
    ).strip()

    target_column = input(
        "\nEnter target column name: "
    ).strip()

    recommend_model(
        dataset_path,
        target_column
    )


if __name__ == "__main__":
    main()