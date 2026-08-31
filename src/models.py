"""
ML Model Selection Advisor
==========================

Model definitions and preprocessing pipelines.

Supported models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. KNN
    5. SVM
    6. XGBoost

All preprocessing is performed inside sklearn Pipelines so that
cross-validation does not suffer from data leakage.
"""

from __future__ import annotations
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ==============================================================
# CONFIGURATION
# ==============================================================

RANDOM_STATE = 42


# ==============================================================
# FEATURE DETECTION
# ==============================================================

def _get_feature_columns(X: pd.DataFrame):
    """
    Detect numeric and categorical feature columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    numeric_features : list
    categorical_features : list
    """

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


# ==============================================================
# VERSION-COMPATIBLE ONE-HOT ENCODER
# ==============================================================

def _make_one_hot_encoder():
    """
    Create a version-compatible OneHotEncoder.

    New sklearn versions use:
        sparse_output=True

    Older versions use:
        sparse=True
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )


# ==============================================================
# PREPROCESSOR
# ==============================================================

def _build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False
):
    """
    Build preprocessing pipeline.

    Numeric features:
        - Median imputation
        - Optional StandardScaler

    Categorical features:
        - Most-frequent imputation
        - One-hot encoding

    Important:
        Imputation and scaling happen INSIDE the pipeline.
        Therefore, each CV training fold learns preprocessing
        only from its training data.
    """

    numeric_features, categorical_features = _get_feature_columns(X)

    transformers = []

    # ----------------------------------------------------------
    # NUMERIC FEATURES
    # ----------------------------------------------------------

    if numeric_features:

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]

        if scale_numeric:

            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

        numeric_pipeline = Pipeline(
            steps=numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # ----------------------------------------------------------
    # CATEGORICAL FEATURES
    # ----------------------------------------------------------

    if categorical_features:

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
                    _make_one_hot_encoder()
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    # ----------------------------------------------------------
    # COLUMN TRANSFORMER
    # ----------------------------------------------------------

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ==============================================================
# MODEL FACTORY
# ==============================================================

def get_models(X: pd.DataFrame):
    """
    Return all six ML model pipelines.

    Parameters
    ----------
    X : pd.DataFrame
        Dataset feature dataframe.

    Returns
    -------
    dict
        Dictionary containing six sklearn pipelines.
    """

    models = {}

    # ==========================================================
    # 1. LOGISTIC REGRESSION
    # ==========================================================

    logistic_preprocessor = _build_preprocessor(
        X,
        scale_numeric=True
    )

    models["logistic_regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                logistic_preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    # ==========================================================
    # 2. DECISION TREE
    # ==========================================================

    tree_preprocessor = _build_preprocessor(
        X,
        scale_numeric=False
    )

    models["decision_tree"] = Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor
            ),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    # ==========================================================
    # 3. RANDOM FOREST
    # ==========================================================

    rf_preprocessor = _build_preprocessor(
        X,
        scale_numeric=False
    )

    models["random_forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                rf_preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )
            )
        ]
    )

    # ==========================================================
    # 4. KNN
    # ==========================================================

    knn_preprocessor = _build_preprocessor(
        X,
        scale_numeric=True
    )

    models["knn"] = Pipeline(
        steps=[
            (
                "preprocessor",
                knn_preprocessor
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]
    )

    # ==========================================================
    # 5. SVM
    # ==========================================================

    svm_preprocessor = _build_preprocessor(
        X,
        scale_numeric=True
    )

    models["svm"] = Pipeline(
        steps=[
            (
                "preprocessor",
                svm_preprocessor
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    # ==========================================================
    # 6. XGBOOST
    # ==========================================================

    xgb_preprocessor = _build_preprocessor(
        X,
        scale_numeric=False
    )

    models["xgboost"] = Pipeline(
        steps=[
            (
                "preprocessor",
                xgb_preprocessor
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
                    n_jobs=1,
                    eval_metric="logloss"
                )
            )
        ]
    )

    return models