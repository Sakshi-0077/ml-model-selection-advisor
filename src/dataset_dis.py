import openml
import pandas as pd
def main():
    suite = openml.study.get_suite("OpenML-CC18")
    rows = []
    for task_id in suite.tasks:
        print(f"Getting metadata for task {task_id}...")
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = task.get_dataset(
                download_data=False,
                download_qualities=True,
                download_features_meta_data=False
            )
            qualities = dataset.qualities
            row = {
                "task_id": task_id,
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.name,
                "target_column": task.target_name,
                "n_samples": qualities.get("NumberOfInstances"),
                "n_features": qualities.get("NumberOfFeatures"),
                "n_classes": qualities.get("NumberOfClasses")
            }
            rows.append(row)
        except Exception as e:
            print(f"Skipping task {task_id}: {e}")
    df = pd.DataFrame(rows)
    print("\nDataset metadata:")
    print(df.to_string(index=False))
    output_file = "data/processed/openml_cc18_metadata.csv"
    df.to_csv(output_file, index=False)
    print(f"\nMetadata saved to: {output_file}")
    print(f"Datasets collected: {len(df)}")

if __name__ == "__main__":
    main()