import pandas as pd

bank = pd.read_csv("data/clean/bank_clean.csv")

qis = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]

for col in qis:
    print("=" * 60)
    print(f"ATTRIBUTE: {col}")
    print(f"Type: {'numeric' if bank[col].dtype in ['int64', 'float64'] else 'categorical'}")
    print(f"Unique values: {bank[col].nunique()}")
    print("-" * 60)

    if bank[col].dtype in ['int64', 'float64']:
        print(f"Min: {bank[col].min()}, Max: {bank[col].max()}, Mean: {bank[col].mean():.1f}")
        print("\nDistribution (10 bins):")
        print(bank[col].value_counts(bins=10, sort=False))
    else:
        print("\nValue counts:")
        vc = bank[col].value_counts()
        for val, count in vc.items():
            pct = 100 * count / len(bank)
            print(f"  {val:<30} {count:>6} ({pct:>5.1f}%)")

    print("\n")