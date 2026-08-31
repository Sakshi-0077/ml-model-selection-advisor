import openml
import pandas as pd
import numpy as np

def profile_dataset(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, feature_names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe")
    numeric_count = len(categorical_indicator) - sum(categorical_indicator)
    categorical_count = sum(categorical_indicator)
    class_counts = y.value_counts()
    class_probabilities = class_counts / len(y)
    target_entropy = -np.sum(class_probabilities * np.log2(class_probabilities))
    imbalance_ratio = class_counts.min() / class_counts.max()
    numeric_columns = X.select_dtypes(include="number").columns

    if len(numeric_columns) > 0:
        numeric_data = X[numeric_columns]
        mean_feature_mean = numeric_data.mean().mean()
        mean_feature_std = numeric_data.std().mean()
        mean_feature_variance = numeric_data.var().mean()
        correlation_matrix = numeric_data.corr()
        correlation_values = correlation_matrix.values
        upper_triangle = correlation_values[np.triu_indices_from(correlation_values, k=1)]
        if len(upper_triangle) == 0:
            mean_abs_correlation = 0.0
        else:
            mean_abs_correlation = np.nanmean(np.abs(upper_triangle))
    else:
        mean_feature_mean = 0.0
        mean_feature_std = 0.0
        mean_feature_variance = 0.0
        mean_abs_correlation = 0.0
    profile = {
        "task_id": task_id,
        "dataset_name": dataset.name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_numeric": numeric_count,
        "n_categorical": categorical_count,
        "missing_percentage": X.isna().mean().mean() * 100,
        "n_classes": y.nunique(),
        "imbalance_ratio": imbalance_ratio,
        "target_entropy": target_entropy,
        "mean_feature_mean": mean_feature_mean,
        "mean_feature_std": mean_feature_std,
        "mean_feature_variance": mean_feature_variance,
        "mean_abs_correlation": mean_abs_correlation,
    }
    return profile

if __name__ == "__main__":
    result = profile_dataset(29)
    for key, value in result.items():
        print(f"{key}: {value}")