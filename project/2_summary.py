import pandas as pd

datasets = {
    "adult": ("data/raw/adult.csv", "income"),
    "bank": ("data/raw/bank.csv", "y"),
    "german": ("data/raw/german.csv", "class")
}

for name, (path, target) in datasets.items():
    df = pd.read_csv(path)

    print("=" * 60)
    print(f"DATASET: {name.upper()}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target: '{target}'")
    print("=" * 60)

    # Target distribution
    print(f"\nTarget Distribution:")
    print(df[target].value_counts(normalize=True).round(3))

    # Column info
    print(f"\nColumns:")
    print("-" * 60)
    print(f"{'Column':<25} {'Type':<10} {'Unique':<8} {'Missing':<10}")
    print("-" * 60)

    for col in df.columns:
        dtype = "num" if df[col].dtype in ['int64', 'float64'] else "cat"
        unique = df[col].nunique()
        missing = df[col].isna().sum() + (df[col] == "?").sum()
        missing_pct = f"{missing} ({100 * missing / len(df):.1f}%)" if missing > 0 else "0"
        print(f"{col:<25} {dtype:<10} {unique:<8} {missing_pct:<10}")

    print("\n")