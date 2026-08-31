from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import (
    SVC,
    SVR,
)

from sklearn.utils.multiclass import type_of_target

from advisor import (
    recommend_model,
    detect_task_type,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "final_model.pkl"
)

METADATA_PATH = (
    MODELS_DIR
    / "model_metadata.pkl"
)


# =========================================================
# MODEL FACTORY
# =========================================================

def get_model(
    model_name,
    task_type
):

    name = (
        str(model_name)
        .strip()
        .lower()
    )

    aliases = {

        "logistic":
            "logistic regression",

        "logistic regression":
            "logistic regression",

        "decisiontree":
            "decision tree",

        "decision tree":
            "decision tree",

        "randomforest":
            "random forest",

        "random forest":
            "random forest",

        "gradientboosting":
            "gradient boosting",

        "gradient boosting":
            "gradient boosting",

        "knn":
            "knn",

        "k-nearest neighbors":
            "knn",

        "k nearest neighbors":
            "knn",

        "naive bayes":
            "naive bayes",

        "gaussian nb":
            "naive bayes",

        "svm":
            "svm",

        "support vector machine":
            "svm",

        "linear regression":
            "linear regression",

        "ridge":
            "ridge",

    }

    normalized_name = aliases.get(
        name,
        name
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if task_type == "classification":

        models = {

            "logistic regression":
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                ),

            "decision tree":
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10
                ),

            "random forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=15
                ),

            "gradient boosting":
                GradientBoostingClassifier(
                    random_state=42
                ),

            "knn":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

            "naive bayes":
                GaussianNB(),

            "svm":
                SVC(
                    probability=True,
                    random_state=42
                ),
        }

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        models = {

            "linear regression":
                LinearRegression(),

            "ridge":
                Ridge(
                    alpha=1.0
                ),

            "decision tree":
                GradientBoostingRegressor(
                    random_state=42
                ),

            "random forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=15
                ),

            "gradient boosting":
                GradientBoostingRegressor(
                    random_state=42
                ),

            "knn":
                KNeighborsRegressor(
                    n_neighbors=5
                ),

            "svm":
                SVR(),
        }

    if normalized_name not in models:

        raise ValueError(
            f"\nUnsupported model '{model_name}' "
            f"for {task_type}.\n\n"
            f"Supported models:\n"
            f"{list(models.keys())}"
        )

    return models[
        normalized_name
    ]


# =========================================================
# PREPROCESSOR
# =========================================================

def build_preprocessor(X):

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    print(
        "\nNumeric columns:"
    )

    print(
        numeric_columns
    )

    print(
        "\nCategorical columns:"
    )

    print(
        categorical_columns
    )

    transformers = []

    # -----------------------------------------------------
    # Numeric
    # -----------------------------------------------------

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[

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
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    # -----------------------------------------------------
    # Categorical
    # -----------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if not transformers:

        raise ValueError(
            "No usable feature columns found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# =========================================================
# REMOVE IDENTIFIER COLUMNS
# =========================================================

def remove_identifier_columns(X):

    identifier_columns = []

    for column in X.columns:

        name = (
            str(column)
            .strip()
            .lower()
        )

        unique_ratio = (
            X[column]
            .nunique(dropna=False)
            / max(len(X), 1)
        )

        identifier_name = (
            name in [
                "id",
                "index",
                "record_id",
                "customer_id",
                "user_id",
                "row_id"
            ]
            or name.endswith("_id")
            or name.endswith("id")
        )

        if (
            identifier_name
            and unique_ratio > 0.80
        ):

            identifier_columns.append(
                column
            )

    if identifier_columns:

        print(
            "\nIdentifier columns removed:"
        )

        print(
            identifier_columns
        )

        X = X.drop(
            columns=identifier_columns
        )

    return X, identifier_columns


# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline(
    X,
    model_name,
    task_type
):

    preprocessor = (
        build_preprocessor(X)
    )

    model = get_model(
        model_name,
        task_type
    )

    return Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            ),
        ]
    )


# =========================================================
# TRAIN FINAL MODEL
# =========================================================

def train_final_model(
    dataset_path,
    target_column
):

    print(
        "\n" + "=" * 70
    )

    print(
        "END-TO-END FINAL MODEL TRAINING"
    )

    print(
        "=" * 70
    )

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"\nCSV file not found:\n"
            f"{dataset_path}"
        )

    # =====================================================
    # LOAD
    # =====================================================

    df = pd.read_csv(
        dataset_path
    )

    if df.empty:

        raise ValueError(
            "The CSV file is empty."
        )

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    print(
        f"\nDataset: {dataset_path}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # =====================================================
    # REMOVE MISSING TARGET
    # =====================================================

    df = df.dropna(
        subset=[target_column]
    )

    if df.empty:

        raise ValueError(
            "No usable rows remain."
        )

    # =====================================================
    # X / Y
    # =====================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # =====================================================
    # TASK DETECTION
    # =====================================================

    task_type = detect_task_type(
        y
    )

    print(
        f"\nTask type: {task_type}"
    )

    if task_type == "classification":

        print(
            f"Classes: {y.nunique()}"
        )

        print(
            "\nClass distribution:"
        )

        print(
            y.value_counts()
        )

    else:

        print(
            f"Target minimum: {y.min()}"
        )

        print(
            f"Target maximum: {y.max()}"
        )

        print(
            f"Target mean: {y.mean():.4f}"
        )

    # =====================================================
    # REMOVE IDENTIFIER FEATURES
    # =====================================================

    X, removed_identifier_columns = (
        remove_identifier_columns(X)
    )

    # =====================================================
    # ADVISOR
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ASKING MODEL SELECTION ADVISOR"
    )

    print(
        "=" * 70
    )

    advisor_details = recommend_model(
        dataset_path,
        target_column,
        return_details=True
    )

    recommended_model = (
        advisor_details[
            "recommended_model"
        ]
    )

    # =====================================================
    # SAFETY CHECK
    # =====================================================

    # Never allow a classifier on regression data.
    # Never allow regression model on classification data.

    if task_type == "regression":

        allowed_models = [
            "linear regression",
            "ridge",
            "random forest",
            "gradient boosting",
            "knn",
            "svm",
            "decision tree"
        ]

        if (
            recommended_model.lower()
            not in allowed_models
        ):

            recommended_model = (
                "Gradient Boosting"
            )

    else:

        allowed_models = [
            "logistic regression",
            "decision tree",
            "random forest",
            "gradient boosting",
            "knn",
            "naive bayes",
            "svm"
        ]

        if (
            recommended_model.lower()
            not in allowed_models
        ):

            recommended_model = (
                "Random Forest"
            )

    print(
        f"\nADVISOR RECOMMENDS: "
        f"{recommended_model}"
    )

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    stratify = None

    if task_type == "classification":

        class_counts = (
            y.value_counts()
        )

        if (
            len(class_counts) > 1
            and class_counts.min() >= 2
        ):

            stratify = y

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAIN / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # =====================================================
    # EVALUATION MODEL
    # =====================================================

    evaluation_pipeline = (
        build_pipeline(
            X_train,
            recommended_model,
            task_type
        )
    )

    print(
        "\nTraining evaluation model..."
    )

    evaluation_pipeline.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICT
    # =====================================================

    y_pred = (
        evaluation_pipeline.predict(
            X_test
        )
    )

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================

    if task_type == "classification":

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "HELD-OUT TEST EVALUATION"
        )

        print(
            "=" * 70
        )

        print(
            f"\nAccuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Weighted F1 Score: "
            f"{f1:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        metrics = {

            "accuracy":
                float(accuracy),

            "balanced_accuracy":
                float(balanced_accuracy),

            "f1_score":
                float(f1),

        }

    # =====================================================
    # REGRESSION METRICS
    # =====================================================

    else:

        r2 = r2_score(
            y_test,
            y_pred
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "HELD-OUT TEST EVALUATION"
        )

        print(
            "=" * 70
        )

        print(
            f"\nR2 Score: "
            f"{r2:.4f}"
        )

        print(
            f"MAE: "
            f"{mae:.4f}"
        )

        print(
            f"RMSE: "
            f"{rmse:.4f}"
        )

        metrics = {

            "r2_score":
                float(r2),

            "mae":
                float(mae),

            "rmse":
                float(rmse),

        }

    # =====================================================
    # FINAL MODEL 100%
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING FINAL MODEL ON 100% OF DATA"
    )

    print(
        "=" * 70
    )

    final_pipeline = (
        build_pipeline(
            X,
            recommended_model,
            task_type
        )
    )

    print(
        "\nTraining on complete dataset..."
    )

    final_pipeline.fit(
        X,
        y
    )

    print(
        "100% training completed successfully."
    )

    # =====================================================
    # CREATE MODEL DIRECTORY
    # =====================================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    with open(
        FINAL_MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            final_pipeline,
            file
        )

    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "dataset":
            str(dataset_path),

        "target_column":
            target_column,

        "task_type":
            task_type,

        "recommended_model":
            recommended_model,

        "feature_columns":
            list(X.columns),

        "removed_identifier_columns":
            removed_identifier_columns,

        "training_samples":
            len(X),

        "evaluation_samples":
            len(X_test),

        **metrics,
    }

    with open(
        METADATA_PATH,
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file
        )

    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL SAVED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModel:\n"
        f"{FINAL_MODEL_PATH}"
    )

    print(
        f"\nMetadata:\n"
        f"{METADATA_PATH}"
    )

    print(
        "\nTask type:"
    )

    print(
        task_type
    )

    print(
        "\nModel:"
    )

    print(
        recommended_model
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "END-TO-END TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    return {
        "model_path":
            FINAL_MODEL_PATH,

        "metadata_path":
            METADATA_PATH,

        "task_type":
            task_type,

        "model":
            recommended_model,

        "metrics":
            metrics,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "FINAL MODEL TRAINER"
    )

    print(
        "=" * 70
    )

    dataset_path = input(
        "\nEnter CSV file path: "
    ).strip()

    target_column = input(
        "Enter target column name: "
    ).strip()

    train_final_model(
        dataset_path,
        target_column
    )


if __name__ == "__main__":

    main()