from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/cleaned/selector_training_data.csv")
MODEL_PATH = Path("data/processed/model_selector.pkl")
OUTPUT_PATH = Path("data/processed/selector_evaluation.csv")

print("=" * 70)
print("EVALUATING MODEL SELECTION ADVISOR")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

best_idx = df.groupby("dataset_name")["f1_score"].idxmax()
best_models = df.loc[best_idx].copy()

selector_data = joblib.load(MODEL_PATH)

selector = selector_data["model"]
features = selector_data["features"]

X = best_models[features]
y = best_models["model"]

predictions = selector.predict(X)

accuracy = accuracy_score(y, predictions)

print(f"\nDatasets evaluated: {len(X)}")
print(f"Selector accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

evaluation = best_models[
    [
        "dataset_name",
        "n_samples",
        "n_features",
        "n_classes",
        "model",
    ]
].copy()

evaluation["predicted_model"] = predictions
evaluation["correct"] = (
    evaluation["model"] == evaluation["predicted_model"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
evaluation.to_csv(OUTPUT_PATH, index=False)

print("\nActual distribution:")
print(y.value_counts())

print("\nPredicted distribution:")
print(pd.Series(predictions).value_counts())

print("\nCorrect predictions:")
print(evaluation["correct"].sum())

print("Incorrect predictions:")
print((~evaluation["correct"]).sum())

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)