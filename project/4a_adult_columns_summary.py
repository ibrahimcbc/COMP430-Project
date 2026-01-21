import pandas as pd

adult = pd.read_csv("data/clean/adult_clean.csv")

# All QIs (public + private, excluding target)
qis = [
    "age", "workclass", "education", "marital-status", "occupation",
    "race", "sex", "native-country", "relationship",
    "capital-gain", "capital-loss", "hours-per-week"
]

for col in qis:
    print("=" * 60)
    print(f"ATTRIBUTE: {col}")
    print(f"Type: {'numeric' if adult[col].dtype in ['int64', 'float64'] else 'categorical'}")
    print(f"Unique values: {adult[col].nunique()}")
    print("-" * 60)

    if adult[col].dtype in ['int64', 'float64']:
        # Numeric: show stats and binned distribution
        print(f"Min: {adult[col].min()}, Max: {adult[col].max()}, Mean: {adult[col].mean():.1f}")
        print("\nDistribution (10 bins):")
        print(adult[col].value_counts(bins=10, sort=False))
    else:
        # Categorical: show all values with counts
        print("\nValue counts:")
        vc = adult[col].value_counts()
        for val, count in vc.items():
            pct = 100 * count / len(adult)
            print(f"  {val:<30} {count:>6} ({pct:>5.1f}%)")

    print("\n")