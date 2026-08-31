import time

import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from models import get_models


def main():
    task = openml.tasks.get_task(37)

    X, y = task.get_X_and_y(
        dataset_format="dataframe"
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    models = get_models(X)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():
        print(f"Running {name}...")

        start = time.perf_counter()

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        elapsed = time.perf_counter() - start

        results.append({
            "dataset": "diabetes",
            "model": name,
            "f1_macro_mean": scores.mean(),
            "f1_macro_std": scores.std(),
            "training_time": elapsed
        })

        print(
            f"  F1: {scores.mean():.4f} "
            f"(±{scores.std():.4f})"
        )

    results_df = pd.DataFrame(results)

    print("\nResults:")
    print(results_df.to_string(index=False))

    output_file = "data/processed/benchmark_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()