from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT_PATH = Path("data/cleaned/selector_training_data.csv")
MODEL_PATH = Path("data/processed/model_selector.pkl")

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("TRAINING MODEL SELECTION ADVISOR")
print("=" * 70)

best_idx = df.groupby("dataset_name")["f1_score"].idxmax()
best_models = df.loc[best_idx].copy()

features = [
    "n_samples",
    "n_features",
    "n_classes",
    "numeric_features",
    "categorical_features",
    "missing_percentage",
]

X = best_models[features]
y = best_models["model"]

print(f"\nTraining datasets: {len(best_models)}")
print(f"Features: {features}")

print("\nTarget distribution:")
print(y.value_counts())

# Train on all 68 datasets
selector = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

selector.fit(X, y)

predictions = selector.predict(X)
accuracy = accuracy_score(y, predictions)

print("\n" + "=" * 70)
print("MODEL SELECTOR RESULTS")
print("=" * 70)

print(f"\nTraining accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": selector,
        "features": features,
    },
    MODEL_PATH
)

print("=" * 70)
print("MODEL SELECTOR SAVED")
print("=" * 70)
print(MODEL_PATH)