import pandas as pd

german = pd.read_csv("data/clean/german_clean.csv")

qis = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status", "other_parties", "residence_since", "property",
    "age", "other_payment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker"
]

for col in qis:
    print("=" * 60)
    print(f"ATTRIBUTE: {col}")
    print(f"Type: {'numeric' if german[col].dtype in ['int64', 'float64'] else 'categorical'}")
    print(f"Unique values: {german[col].nunique()}")
    print("-" * 60)

    if german[col].dtype in ['int64', 'float64']:
        print(f"Min: {german[col].min()}, Max: {german[col].max()}, Mean: {german[col].mean():.1f}")
        print("\nDistribution (10 bins):")
        print(german[col].value_counts(bins=10, sort=False))
    else:
        print("\nValue counts:")
        vc = german[col].value_counts()
        for val, count in vc.items():
            pct = 100 * count / len(german)
            print(f"  {val:<30} {count:>6} ({pct:>5.1f}%)")

    print("\n")