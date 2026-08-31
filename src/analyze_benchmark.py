from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "benchmark_clean.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BENCHMARK ANALYSIS")
print("=" * 70)

datasets = df.drop_duplicates("dataset_name").copy()

print(f"\nDatasets: {len(datasets)}")
print(f"Models: {df['model'].nunique()}")
print(f"Benchmark rows: {len(df)}")

print("\nDataset characteristics:")
print(
    datasets[
        [
            "dataset_name",
            "n_samples",
            "n_features",
            "n_classes",
            "numeric_features",
            "categorical_features",
            "missing_percentage",
        ]
    ].to_string(index=False)
)

print("\nMissing values:")
print(
    datasets[
        [
            "n_samples",
            "n_features",
            "n_classes",
            "numeric_features",
            "categorical_features",
            "missing_percentage",
        ]
    ].isna().sum()
)

print("\nAverage performance by model:")
model_summary = (
    df.groupby("model")
    .agg(
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        f1_score=("f1_score", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        training_time=("training_time", "mean"),
    )
    .sort_values("accuracy", ascending=False)
)

print(model_summary.round(4).to_string())

print("\nBest model for each dataset:")

best_models = (
    df.sort_values(
        ["dataset_name", "f1_score"],
        ascending=[True, False]
    )
    .groupby("dataset_name")
    .first()
    .reset_index()
)

print(
    best_models[
        [
            "dataset_name",
            "model",
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "training_time",
        ]
    ].to_string(index=False)
)

print("\nBest-model distribution:")

print(
    best_models["model"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)