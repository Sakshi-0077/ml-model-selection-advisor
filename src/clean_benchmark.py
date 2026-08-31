from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "benchmark_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_clean.csv"

print("=" * 70)
print("CLEANING BENCHMARK RESULTS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")

columns_to_keep = [
    "task_id",
    "dataset_id",
    "dataset_name",
    "target_column",
    "model",
    "n_samples",
    "n_features",
    "n_classes",
    "numeric_features",
    "categorical_features",
    "missing_percentage",
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "training_time",
    "status",
    "error",
    "data_source",
]

df = df[columns_to_keep].copy()

df = df.dropna(subset=["dataset_name", "model"])

numeric_columns = [
    "n_samples",
    "n_features",
    "n_classes",
    "numeric_features",
    "categorical_features",
    "missing_percentage",
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "training_time",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "training_time",
]

df["metrics_complete"] = df[metric_columns].notna().all(axis=1)

duplicates = df.duplicated(
    subset=["dataset_id", "model"],
    keep=False
)

print(f"Duplicate combinations: {duplicates.sum()}")
print(f"Cleaned shape: {df.shape}")
print(f"Unique datasets: {df['dataset_name'].nunique()}")
print(f"Unique models: {df['model'].nunique()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nCleaning complete.")