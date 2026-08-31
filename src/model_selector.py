from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("data/processed/model_selector.pkl")

selector_data = joblib.load(MODEL_PATH)

model = selector_data["model"]
features = selector_data["features"]


def recommend_model(
    n_samples,
    n_features,
    n_classes,
    numeric_features,
    categorical_features,
    missing_percentage
):
    data = pd.DataFrame([{
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_percentage": missing_percentage
    }])

    prediction = model.predict(data[features])[0]

    probabilities = model.predict_proba(data[features])[0]

    recommendations = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "recommended_model": prediction,
        "recommendations": [
            {
                "model": name,
                "confidence": round(float(prob), 4)
            }
            for name, prob in recommendations
        ]
    }


if __name__ == "__main__":

    result = recommend_model(
        n_samples=5000,
        n_features=20,
        n_classes=2,
        numeric_features=20,
        categorical_features=0,
        missing_percentage=0
    )

    print("=" * 70)
    print("MODEL SELECTION ADVISOR")
    print("=" * 70)

    print("\nRecommended model:")
    print(result["recommended_model"])

    print("\nModel ranking:")
    for item in result["recommendations"]:
        print(
            f"{item['model']}: "
            f"{item['confidence']:.2%}"
        )