import openml

from models import get_models


TASKS = [37, 3, 29]


def main():
    for task_id in TASKS:
        print(f"\nTesting task {task_id}...")

        task = openml.tasks.get_task(task_id)

        X, y = task.get_X_and_y(
            dataset_format="dataframe"
        )

        models = get_models(X)

        print("Dataset shape:", X.shape)
        print("Models created:", len(models))


if __name__ == "__main__":
    main()