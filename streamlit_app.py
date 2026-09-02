from pathlib import Path
import sys
import tempfile
import pickle

import streamlit as st
import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from advisor import (
    recommend_model,
    detect_task_type,
)

from train_final_model import (
    train_final_model,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ML Model Selection Advisor",
    page_icon=None,
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "ML Model Selection Advisor"
)

st.write(
    "Upload a CSV dataset, select the target column, "
    "analyze the dataset, receive a model recommendation, "
    "train the recommended model, evaluate it, and "
    "download the trained model."
)

st.divider()


# =========================================================
# SESSION STATE
# =========================================================

if "recommended_model" not in st.session_state:

    st.session_state[
        "recommended_model"
    ] = None


if "model_trained" not in st.session_state:

    st.session_state[
        "model_trained"
    ] = False


if "training_result" not in st.session_state:

    st.session_state[
        "training_result"
    ] = None


# =========================================================
# UPLOAD
# =========================================================

st.header(
    "Upload Dataset"
)

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


# =========================================================
# APPLICATION
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV dataset to begin."
    )

    st.stop()


# =========================================================
# READ DATA
# =========================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Could not read CSV file: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded CSV file is empty."
    )

    st.stop()


st.success(
    f"Dataset uploaded successfully: "
    f"{uploaded_file.name}"
)


# =========================================================
# TEMPORARY FILE
# =========================================================

temp_dir = Path(
    tempfile.gettempdir()
)

temp_csv_path = (
    temp_dir
    / uploaded_file.name
)

df.to_csv(
    temp_csv_path,
    index=False
)


# =========================================================
# DATASET INFORMATION
# =========================================================

st.header(
    "Dataset Information"
)

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Rows",
        len(df)
    )

with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )

with col3:

    st.metric(
        "Missing Values",
        f"{df.isna().mean().mean() * 100:.2f}%"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        int(
            df.duplicated().sum()
        )
    )


# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# =========================================================
# TARGET COLUMN
# =========================================================

st.subheader(
    "Select Target Column"
)

target_column = st.selectbox(
    "Choose the column you want to predict:",
    df.columns
)


# =========================================================
# RESET WHEN TARGET CHANGES
# =========================================================

if (
    "previous_target"
    not in st.session_state
):

    st.session_state[
        "previous_target"
    ] = target_column


if (
    st.session_state[
        "previous_target"
    ] != target_column
):

    st.session_state[
        "recommended_model"
    ] = None

    st.session_state[
        "model_trained"
    ] = False

    st.session_state[
        "training_result"
    ] = None

    st.session_state[
        "previous_target"
    ] = target_column


# =========================================================
# TARGET TYPE
# =========================================================

try:

    task_type = detect_task_type(
        df[target_column]
    )

except Exception as e:

    st.error(
        f"Could not determine target type: {e}"
    )

    st.stop()


st.subheader(
    "Problem Type"
)

if task_type == "classification":

    st.success(
        "Classification problem"
    )

else:

    st.success(
        "Regression problem"
    )


# =========================================================
# FEATURE INFORMATION
# =========================================================

feature_columns = [
    column
    for column in df.columns
    if column != target_column
]

numeric_columns = (
    df[feature_columns]
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)

categorical_columns = (
    df[feature_columns]
    .select_dtypes(
        exclude=["number"]
    )
    .columns
    .tolist()
)


st.subheader(
    "Feature Information"
)

col1, col2 = st.columns(2)

with col1:

    st.write(
        "**Numeric Features**"
    )

    st.write(
        numeric_columns
    )

with col2:

    st.write(
        "**Categorical Features**"
    )

    st.write(
        categorical_columns
    )


# =========================================================
# TARGET INFORMATION
# =========================================================

st.subheader(
    "Target Information"
)

if task_type == "classification":

    target_classes = (
        df[target_column]
        .nunique()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Number of Classes",
            target_classes
        )

    with col2:

        st.write(
            "**Class Distribution**"
        )

        st.dataframe(
            df[target_column]
            .value_counts(),
            use_container_width=True
        )

else:

    target_numeric = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Minimum",
            f"{target_numeric.min():.4f}"
        )

    with col2:

        st.metric(
            "Maximum",
            f"{target_numeric.max():.4f}"
        )

    with col3:

        st.metric(
            "Mean",
            f"{target_numeric.mean():.4f}"
        )


st.divider()


# =========================================================
# MODEL ADVISOR
# =========================================================

st.header(
    "Model Selection Advisor"
)


if st.button(
    "Get Model Recommendation",
    type="primary"
):

    try:

        with st.spinner(
            "Analyzing dataset..."
        ):

            details = recommend_model(
                temp_csv_path,
                target_column,
                return_details=True
            )

        st.session_state[
            "recommended_model"
        ] = details[
            "recommended_model"
        ]

        st.session_state[
            "advisor_details"
        ] = details

        st.session_state[
            "model_trained"
        ] = False

        st.session_state[
            "training_result"
        ] = None

        st.success(
            "Model recommendation completed."
        )

    except Exception as e:

        st.error(
            f"Model recommendation failed: {e}"
        )


# =========================================================
# DISPLAY RECOMMENDATION
# =========================================================

if (
    st.session_state[
        "recommended_model"
    ]
    is not None
):

    recommended_model = (
        st.session_state[
            "recommended_model"
        ]
    )

    st.subheader(
        "Recommended Model"
    )

    st.success(
        recommended_model
    )

    # -----------------------------------------------------
    # Advisor details
    # -----------------------------------------------------

    details = st.session_state.get(
        "advisor_details",
        {}
    )

    if details:

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Samples",
                details.get(
                    "samples",
                    "-"
                )
            )

        with col2:

            st.metric(
                "Features",
                details.get(
                    "features",
                    "-"
                )
            )

        with col3:

            st.metric(
                "Task",
                details.get(
                    "task_type",
                    "-"
                )
            )

    # =====================================================
    # TRAIN
    # =====================================================

    st.divider()

    st.header(
        "Train Recommended Model"
    )

    st.write(
        f"The advisor recommends "
        f"**{recommended_model}** "
        f"for this dataset."
    )

    if st.button(
        "Train Model",
        type="primary"
    ):

        try:

            with st.spinner(
                f"Training {recommended_model}..."
            ):

               result = train_final_model(
                temp_csv_path,
                target_column
            )

            st.session_state["model_trained"] = True
            st.session_state["training_result"] = result
            st.session_state["model_path"] = str(result["model_path"])
            st.session_state["metadata_path"] = str(result["metadata_path"])

            st.success(
                "Model trained successfully."
            )

        except Exception as e:

            st.session_state[
                "model_trained"
            ] = False

            st.error(
                "Model training failed."
            )

            st.exception(e)


# =========================================================
# TRAINING RESULTS
# =========================================================

if st.session_state.get(
    "model_trained",
    False
):

    st.divider()

    st.header(
        "Training Results"
    )

    result = st.session_state.get(
        "training_result"
    )

    model_path = Path(
        st.session_state["model_path"]
    )

    metadata_path = Path(
        st.session_state["metadata_path"]
    )

    # =====================================================
    # LOAD METADATA
    # =====================================================

    if metadata_path.exists():

        with open(
            metadata_path,
            "rb"
        ) as file:

            metadata = pickle.load(
                file
            )

        # -------------------------------------------------
        # MODEL
        # -------------------------------------------------

        st.write(
            f"**Model:** "
            f"{metadata.get('recommended_model', '-')}"
        )

        st.write(
            f"**Problem Type:** "
            f"{metadata.get('task_type', '-')}"
        )

        # =================================================
        # CLASSIFICATION RESULTS
        # =================================================

        if metadata.get(
            "task_type"
        ) == "classification":

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "Accuracy",
                    f"{metadata.get('accuracy', 0) * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Balanced Accuracy",
                    f"{metadata.get('balanced_accuracy', 0) * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Weighted F1",
                    f"{metadata.get('f1_score', 0) * 100:.2f}%"
                )

        # =================================================
        # REGRESSION RESULTS
        # =================================================

        else:

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                st.metric(
                    "R2 Score",
                    f"{metadata.get('r2_score', 0):.4f}"
                )

            with col2:

                st.metric(
                    "MAE",
                    f"{metadata.get('mae', 0):.4f}"
                )

            with col3:

                st.metric(
                    "RMSE",
                    f"{metadata.get('rmse', 0):.4f}"
                )

    # =========================================================
    # DOWNLOAD TRAINED MODEL
    # =========================================================

    st.divider()

    st.header("Download Trained Model")

    if model_path.exists():

        with open(model_path, "rb") as file:
            model_bytes = file.read()

        st.success("Trained model is ready for download.")

        st.download_button(
            label="Download Trained Model",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream",
            key="download_model"
        )

    else:

        st.warning(
            "Trained model file was not found. "
            "Please train the model again."
        )

        st.write(
            f"Expected model location: `{model_path}`"
        )

    # =====================================================
    # DOWNLOAD METADATA
    # =====================================================

    if metadata_path.exists():

        with open(
            metadata_path,
            "rb"
        ) as file:

            metadata_bytes = file.read()

        st.download_button(
            label="Download Model Metadata",
            data=metadata_bytes,
            file_name="model_metadata.pkl",
            mime="application/octet-stream"
        )
# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <style>
    .github-footer {
        position: fixed;
        bottom: 15px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
        z-index: 9999;
        font-size: 14px;
    }

    .github-footer img {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        transition: transform 0.2s;
    }

    .github-footer img:hover {
        transform: scale(1.15);
    }

    .github-footer .ampersand {
        font-size: 18px;
        font-weight: bold;
    }
    </style>

    <div class="github-footer">
        <span>Created by</span>
          <a href="https://github.com/Sakshi-0077" target="_blank">
                    <img
                        src="https://github.com/Sakshi-0077.png"
                        alt="Sakshi"
                    >
                </a>
        <span class="ampersand">&amp;</span>
        <a href="https://github.com/yashvi-02" target="_blank">
                   <img
                       src="https://github.com/yashvi-02.png"
                       alt="Yashvi"
                   >
               </a>
    </div>
    """
)