import openml
import pandas as pd

TASK_ID = 3

def main():
    task = openml.tasks.get_task(TASK_ID)
    dataset = task.get_dataset()
    X, y, categorical_indicator, feature_names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe")
    print("Dataset:", dataset.name)
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])
    print("Target:", task.target_name)
    print("\nFeature types:")
    print(X.dtypes.value_counts())
    print("\nMissing values:")
    print(X.isna().sum().sum())
    print("\nCategorical features:", sum(categorical_indicator))
    print("Numerical features:", len(categorical_indicator) - sum(categorical_indicator))
    print("\nTarget distribution:")
    print(y.value_counts())

if __name__ == "__main__":
    main()