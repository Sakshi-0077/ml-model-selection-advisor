import joblib
import pandas as pd

MODEL_PATH = "data/processed/model_selector.pkl"

selector_data = joblib.load(MODEL_PATH)

print("Saved object type:", type(selector_data))

if isinstance(selector_data, dict):
    print("Dictionary keys:", selector_data.keys())

    selector = (
        selector_data.get("model")
        or selector_data.get("selector")
        or selector_data.get("classifier")
    )

    if selector is None:
        for value in selector_data.values():
            if hasattr(value, "predict"):
                selector = value
                break

    if selector is None:
        raise ValueError("Could not find trained model inside model_selector.pkl")
else:
    selector = selector_data

test_data = pd.DataFrame([
    {
        "n_samples": 5000,
        "n_features": 20,
        "n_classes": 2,
        "numeric_features": 20,
        "categorical_features": 0,
        "missing_percentage": 0
    }
])

prediction = selector.predict(test_data)[0]

print("=" * 70)
print("MODEL SELECTION TEST")
print("=" * 70)

print("\nRecommended model:")
print(prediction)

if hasattr(selector, "predict_proba"):
    probabilities = selector.predict_proba(test_data)[0]
    models = selector.classes_

    print("\nModel ranking:")

    ranking = sorted(
        zip(models, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    for model, probability in ranking:
        print(f"{model}: {probability * 100:.2f}%")