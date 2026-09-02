# ML Model Selection Advisor

An end-to-end machine learning application that analyzes a user-provided dataset, automatically identifies the problem type, recommends a suitable machine learning algorithm, trains the recommended model, evaluates its performance, and provides the trained model for download.

The project combines a **Python-based ML backend** with an interactive **Streamlit interface**, making model selection and training accessible without requiring users to manually experiment with multiple algorithms.

## Live Demo

**Streamlit App:**
https://ml-model-selection-advisor-xf92lufeh2jxm9w8qlzuar.streamlit.app/

## Features

* Upload CSV datasets directly through the web interface
* Automatic dataset analysis
* Automatic classification/regression detection
* Numeric and categorical feature identification
* Target-column selection
* Dataset statistics and class distribution
* Automated machine learning model recommendation
* Training of the recommended model
* Automatic preprocessing for numerical and categorical features
* Missing-value handling
* Identifier-column detection and removal
* Model evaluation using appropriate metrics
* Saving of the trained model using Pickle
* Saving of model metadata
* Download trained model directly from the application
* Download model metadata
* Interactive Streamlit interface

## How It Works

```text
                    ┌─────────────────────┐
                    │    Upload CSV       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Analysis    │
                    │                     │
                    │ • Rows / Columns    │
                    │ • Missing Values    │
                    │ • Features          │
                    │ • Target            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Task Detection      │
                    │                     │
                    │ Classification /   │
                    │ Regression          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Recommendation│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training      │
                    │                     │
                    │ • Preprocessing     │
                    │ • Train/Test Split  │
                    │ • Model Fitting     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Download Model      │
                    │ & Metadata          │
                    └─────────────────────┘
```

## Machine Learning Models

### Classification

The advisor can work with models including:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Support Vector Machine

### Regression

Supported regression approaches include:

* Linear Regression
* Ridge Regression
* Decision Tree / Tree-based regression
* Random Forest Regression
* Gradient Boosting Regression
* K-Nearest Neighbors Regression
* Support Vector Regression

The final model is selected based on the dataset characteristics and detected task type.

## Data Preprocessing

The training pipeline automatically handles common preprocessing requirements.

### Numerical Features

* Missing values → Median imputation
* Feature scaling → StandardScaler

### Categorical Features

* Missing values → Most-frequent imputation
* Encoding → One-Hot Encoding
* Unknown categories → Handled safely using `handle_unknown="ignore"`

The preprocessing and model are combined into a single **Scikit-learn Pipeline**, allowing the complete trained pipeline to be saved and reused.

## Model Evaluation

### Classification

The application reports:

* Accuracy
* Balanced Accuracy
* Weighted F1 Score

### Regression

The application reports:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

## Project Structure

```text
ml-model-selection-advisor/
│
├── advisor.py
├── train_final_model.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── final_model.pkl
│   └── model_metadata.pkl
│
└── data/
```

### Main Components

**`advisor.py`**
Contains the model recommendation and task-detection logic.

**`train_final_model.py`**
Handles preprocessing, model creation, training, evaluation, and model serialization.

**`streamlit_app.py`**
Provides the interactive web interface for dataset upload, model recommendation, training, evaluation, and model download.

**`models/final_model.pkl`**
Stores the complete trained Scikit-learn pipeline.

**`models/model_metadata.pkl`**
Stores information about the selected model, task type, and evaluation metrics.

## Technology Stack

| Category             | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Data Processing      | Pandas, NumPy             |
| Machine Learning     | Scikit-learn              |
| Web Interface        | Streamlit                 |
| Model Serialization  | Pickle                    |
| Deployment           | Streamlit Community Cloud |
| Version Control      | Git & GitHub              |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sakshi-0077/ml-model-selection-advisor.git
cd ml-model-selection-advisor
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

## Usage

1. Open the Streamlit application.
2. Upload a CSV dataset.
3. Select the target column.
4. Review the dataset information.
5. Click **Get Model Recommendation**.
6. Review the recommended machine learning model.
7. Click **Train Model**.
8. Review the evaluation results.
9. Download the trained model.
10. Download the model metadata if required.

## Example Workflow

For a dataset such as the Iris dataset:

```text
CSV Dataset
     ↓
Select "Species" as target
     ↓
Classification detected
     ↓
Dataset analyzed
     ↓
Model recommended
     ↓
Recommended model trained
     ↓
Accuracy / F1 / Balanced Accuracy
     ↓
final_model.pkl generated
     ↓
Download trained model
```

## Model Output

After training, the application generates:

```text
models/
├── final_model.pkl
└── model_metadata.pkl
```

The `final_model.pkl` file contains the complete preprocessing and machine learning pipeline, allowing the trained model to be reused on compatible data.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

https://ml-model-selection-advisor-xf92lufeh2jxm9w8qlzuar.streamlit.app/

## Future Improvements

* Add more machine learning algorithms
* Introduce cross-validation-based model comparison
* Add hyperparameter optimization
* Add feature importance visualization
* Add confusion matrix and ROC-AUC visualization
* Support additional dataset formats
* Add automated model comparison
* Add prediction functionality using uploaded trained models
* Improve model recommendation using advanced dataset profiling
* Add experiment tracking and model versioning

## Contributors

**Sakshi Shah**
Backend & Model Training

**Yashvi Ghaghda**
Streamlit UI & Application Interface

## License

This project is intended for educational and portfolio purposes.
