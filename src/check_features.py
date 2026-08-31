import pandas as pd

FILE = "data/processed/meta_features.csv"

def main():
    df = pd.read_csv(FILE)
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary:")
    print(df.describe())

if __name__ == "__main__":
    main()