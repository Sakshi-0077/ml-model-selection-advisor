import pandas as pd
from profiler import profile_dataset

INPUT_FILE = "data/processed/candidate_datasets.csv"
OUTPUT_FILE = "data/processed/meta_features.csv"

def main():
    datasets = pd.read_csv(INPUT_FILE)
    profiles = []
    for _, row in datasets.iterrows():
        task_id = int(row["task_id"])
        print(f"Profiling {row['dataset_name']} (task {task_id})...")
        try:
            profile = profile_dataset(task_id)
            profiles.append(profile)
        except Exception as e:
            print(f"Skipped {row['dataset_name']}: {e}")
    meta_features = pd.DataFrame(profiles)
    meta_features.to_csv(OUTPUT_FILE, index=False)
    print("\nProfiling completed.")
    print("Datasets successfully profiled:", len(meta_features))
    print("Features generated:", len(meta_features.columns))
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()