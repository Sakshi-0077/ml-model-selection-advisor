from pathlib import Path
import pickle

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.pkl"

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nTrained model not found.\n"
            f"Expected:\n{MODEL_PATH}\n\n"
            "Run train_final_model.py first."
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata():

    if not METADATA_PATH.exists():
        return None

    with open(
        METADATA_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


# =========================================================
# PREDICTION
# =========================================================

def predict(dataset_path):

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\nCSV file not found:\n{dataset_path}"
        )

    print("\nLoading trained model...")

    model = load_model()

    metadata = load_metadata()

    print(
        f"Loading data:\n{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    if df.empty:
        raise ValueError(
            "\nThe CSV file is empty."
        )

    print(
        f"\nSamples: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    if metadata:

        target_column = metadata.get(
            "target_column"
        )

        recommended_model = metadata.get(
            "recommended_model"
        )

        training_features = metadata.get(
            "feature_columns",
            []
        )

        print(
            "\nModel used:"
        )

        print(
            recommended_model
        )

        print(
            "\nExpected features:"
        )

        print(
            training_features
        )

        # =================================================
        # REMOVE TARGET IF PRESENT
        # =================================================

        if target_column in df.columns:

            print(
                f"\nTarget column '{target_column}' "
                "found in input."
            )

            print(
                "Removing target column before prediction."
            )

            df = df.drop(
                columns=[target_column]
            )

    else:

        training_features = list(
            df.columns
        )

    # =====================================================
    # FEATURE VALIDATION
    # =====================================================

    missing_features = [
        column
        for column in training_features
        if column not in df.columns
    ]

    extra_features = [
        column
        for column in df.columns
        if column not in training_features
    ]

    # -----------------------------------------------------
    # Missing required features
    # -----------------------------------------------------

    if missing_features:

        raise ValueError(
            "\nPrediction cannot continue.\n\n"
            "Missing required features:\n"
            f"{missing_features}"
        )

    # -----------------------------------------------------
    # Remove extra columns
    # -----------------------------------------------------

    if extra_features:

        print(
            "\nExtra columns found:"
        )

        print(
            extra_features
        )

        print(
            "\nThese columns will be ignored."
        )

        df = df.drop(
            columns=extra_features
        )

    # =====================================================
    # ENSURE SAME COLUMN ORDER
    # =====================================================

    df = df[
        training_features
    ]

    # =====================================================
    # PREDICT
    # =====================================================

    print(
        "\nMaking predictions..."
    )

    predictions = model.predict(
        df
    )

    # =====================================================
    # RESULT
    # =====================================================

    result = df.copy()

    result[
        "prediction"
    ] = predictions

    # =====================================================
    # PROBABILITY
    # =====================================================

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = (
                model.predict_proba(df)
            )

            result[
                "prediction_probability"
            ] = probabilities.max(
                axis=1
            )

        except Exception:
            pass

    # =====================================================
    # SAVE
    # =====================================================

    output_path = (
        PROJECT_ROOT
        / "predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print(
        f"\nPredictions made: "
        f"{len(predictions)}"
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        pd.Series(predictions)
        .value_counts()
    )

    print(
        f"\nResults saved to:\n"
        f"{output_path}"
    )

    print(
        "\nFirst 10 predictions:"
    )

    print(
        result.head(10)
    )

    print("\n" + "=" * 70)

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("ML MODEL SELECTION ADVISOR - PREDICTION")
    print("=" * 70)

    dataset_path = input(
        "\nEnter new CSV file path: "
    ).strip()

    predict(
        dataset_path
    )


if __name__ == "__main__":
    main()