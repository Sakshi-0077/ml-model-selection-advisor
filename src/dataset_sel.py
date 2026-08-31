import pandas as pd

INPUT_FILE = "data/processed/openml_cc18_metadata.csv"
OUTPUT_FILE = "data/processed/candidate_datasets.csv"

def main():
    df = pd.read_csv(INPUT_FILE)
    candidates = df[
        (df["n_samples"] >= 500) &
        (df["n_features"] >= 2) &
        (df["n_classes"] >= 2)].copy()
    excluded_names = [
        "mnist_784",
        "Fashion-MNIST",
        "CIFAR_10",
        "Devnagari-Script"]
    candidates = candidates[
        ~candidates["dataset_name"].isin(excluded_names)]
    candidates.to_csv(OUTPUT_FILE, index=False)
    print("Total datasets:", len(df))
    print("Candidate datasets:", len(candidates))
    print("\nCandidate datasets:")
    print(
        candidates[
            [
                "task_id",
                "dataset_name",
                "n_samples",
                "n_features",
                "n_classes"
            ]
        ].to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()