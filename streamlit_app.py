import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT BACKEND FUNCTIONS
# ============================================================

from advisor import recommend_model, detect_task_type
from train_final_model import train_final_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Model Selection Advisor",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
            margin-bottom: 35px;
        }

        .recommendation-box {
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #4CAF50;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .metric-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 ML Model Selection Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a dataset and get an intelligent machine learning model recommendation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV dataset from the sidebar to start."
    )

    st.markdown(
        """
        ### How it works

        1. Upload your CSV dataset.
        2. Select the target column.
        3. The system analyzes the dataset.
        4. The ML Model Selection Advisor recommends a model.
        5. The recommended model is trained on your dataset.
        6. The final trained model can be downloaded.

        ### Supported Models

        - XGBoost
        - SVM
        - Random Forest
        - Logistic Regression
        - KNN
        - Decision Tree
        """
    )

else:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")
        st.stop()

    st.success("Dataset uploaded successfully!")


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.header("📊 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ========================================================
    # TARGET COLUMN
    # ========================================================

    st.header("🎯 Target Column")

    target_column = st.selectbox(
        "Select the target column:",
        options=list(df.columns)
    )


    if target_column:

        # ====================================================
        # BASIC DATASET INFORMATION
        # ====================================================

        X = df.drop(columns=[target_column])
        y = df[target_column]

        n_samples = len(df)
        n_features = len(X.columns)
        n_classes = y.nunique()

        numeric_features = X.select_dtypes(
            include=["number"]
        ).shape[1]

        categorical_features = X.select_dtypes(
            exclude=["number"]
        ).shape[1]

        missing_percentage = (
            df.isnull().sum().sum()
            / (df.shape[0] * df.shape[1])
            * 100
        )


        # ====================================================
        # TASK TYPE
        # ====================================================

        try:
            task_type = detect_task_type(y)

        except Exception:
            if y.dtype == "object" or y.nunique() <= 20:
                task_type = "classification"
            else:
                task_type = "regression"


        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        st.header("📋 Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Samples",
                n_samples
            )

        with col2:
            st.metric(
                "Features",
                n_features
            )

        with col3:
            st.metric(
                "Classes",
                n_classes
            )

        with col4:
            st.metric(
                "Missing %",
                f"{missing_percentage:.2f}%"
            )


        # ====================================================
        # FEATURE INFORMATION
        # ====================================================

        st.subheader("Feature Information")

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            st.write(
                f"**Numeric Features:** {numeric_features}"
            )

        with feature_col2:
            st.write(
                f"**Categorical Features:** {categorical_features}"
            )

        st.write(
            f"**Task Type:** {task_type.capitalize()}"
        )


        # ====================================================
        # MODEL RECOMMENDATION
        # ====================================================

        st.header("🧠 Model Recommendation")

        try:

            recommendation = recommend_model(
                df,
                target_column,
                return_details=True
            )

            # Handle dictionary result
            if isinstance(recommendation, dict):

                recommended_model = recommendation.get(
                    "recommended_model",
                    recommendation.get("model")
                )

                ranking = recommendation.get(
                    "ranking",
                    recommendation.get("rankings")
                )

            else:

                recommended_model = recommendation
                ranking = None


            # ------------------------------------------------
            # DISPLAY RECOMMENDATION
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="recommendation-box">

                <h2>🏆 Recommended Model</h2>

                <h1>{recommended_model}</h1>

                <p>
                Based on the dataset characteristics,
                the Model Selection Advisor recommends
                <b>{recommended_model}</b>.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # MODEL RANKING
            # =================================================

            if ranking is not None:

                st.subheader("📈 Model Ranking")

                if isinstance(ranking, dict):

                    ranking_df = pd.DataFrame(
                        list(ranking.items()),
                        columns=["Model", "Score"]
                    )

                    ranking_df = ranking_df.sort_values(
                        by="Score",
                        ascending=False
                    )

                    ranking_df["Score"] = ranking_df[
                        "Score"
                    ].apply(
                        lambda x: f"{x:.2f}%"
                        if isinstance(x, (int, float))
                        else x
                    )

                    st.dataframe(
                        ranking_df,
                        use_container_width=True,
                        hide_index=True
                    )

                elif isinstance(ranking, list):

                    st.write(ranking)


        except Exception as e:

            st.error(
                f"Model recommendation failed: {e}"
            )

            recommended_model = None


        # ====================================================
        # TRAIN FINAL MODEL
        # ====================================================

        if recommended_model:

            st.header("🚀 Train Final Model")

            st.write(
                "The recommended model will now be trained "
                "and evaluated on your uploaded dataset."
            )


            if st.button(
                "Train Recommended Model",
                type="primary"
            ):

                with st.spinner(
                    f"Training {recommended_model}..."
                ):

                    try:

                        # =====================================
                        # CREATE TEMPORARY CSV
                        # =====================================

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".csv"
                        ) as temp_file:

                            temp_file_path = temp_file.name
                            df.to_csv(
                                temp_file_path,
                                index=False
                            )


                        # =====================================
                        # TRAIN FINAL MODEL
                        # =====================================

                        result = train_final_model(
                            temp_file_path,
                            target_column
                        )


                        # =====================================
                        # SUCCESS
                        # =====================================

                        st.success(
                            "Final model trained successfully! 🎉"
                        )


                        # =====================================
                        # DISPLAY RESULT
                        # =====================================

                        if isinstance(result, dict):

                            if "model_name" in result:

                                st.write(
                                    f"**Final Model:** "
                                    f"{result['model_name']}"
                                )

                            if "accuracy" in result:

                                st.metric(
                                    "Accuracy",
                                    f"{result['accuracy']:.4f}"
                                )

                            if "balanced_accuracy" in result:

                                st.metric(
                                    "Balanced Accuracy",
                                    f"{result['balanced_accuracy']:.4f}"
                                )

                            if "f1_score" in result:

                                st.metric(
                                    "F1 Score",
                                    f"{result['f1_score']:.4f}"
                                )


                        # =====================================
                        # MODEL FILES
                        # =====================================

                        model_path = (
                            PROJECT_ROOT
                            / "models"
                            / "final_model.pkl"
                        )

                        metadata_path = (
                            PROJECT_ROOT
                            / "models"
                            / "model_metadata.pkl"
                        )


                        # =====================================
                        # DOWNLOAD FINAL MODEL
                        # =====================================

                        if model_path.exists():

                            with open(
                                model_path,
                                "rb"
                            ) as file:

                                st.download_button(
                                    label="⬇️ Download Final Model",
                                    data=file,
                                    file_name="final_model.pkl",
                                    mime="application/octet-stream"
                                )


                        # =====================================
                        # DOWNLOAD METADATA
                        # =====================================

                        if metadata_path.exists():

                            with open(
                                metadata_path,
                                "rb"
                            ) as file:

                                st.download_button(
                                    label="⬇️ Download Model Metadata",
                                    data=file,
                                    file_name="model_metadata.pkl",
                                    mime="application/octet-stream"
                                )


                    except Exception as e:

                        st.error(
                            f"Model training failed: {e}"
                        )

                    finally:

                        # =====================================
                        # DELETE TEMP FILE
                        # =====================================

                        try:

                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)

                        except Exception:
                            pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <hr>
        <p>
            Created by:
            <b>Yashvi Ghaghda & Sakshi Shah</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)