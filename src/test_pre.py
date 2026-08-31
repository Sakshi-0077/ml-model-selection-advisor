import openml
from preprocessing import build_preprocessor

def main():
    task = openml.tasks.get_task(37)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    print("Original shape:", X.shape)
    print("Processed shape:", X_processed.shape)

if __name__ == "__main__":
    main()