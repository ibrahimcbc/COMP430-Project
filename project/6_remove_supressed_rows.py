import pandas as pd
import os

os.makedirs("data/anonymized_clean", exist_ok=True)

datasets = ["adult", "bank", "german"]
k_values = [2, 5, 10, 20]
strategies = ["public", "all"]

for dataset in datasets:
    for k in k_values:
        for strategy in strategies:
            filename = f"{dataset}_k{k}_{strategy}.csv"

            df = pd.read_csv(f"data/anonymized/{filename}", keep_default_na=False)

            # Check all columns except the last
            feature_cols = df.columns[:-1]
            all_star = df[feature_cols].apply(lambda row: all(str(val) == "*" for val in row), axis=1)
            df = df[~all_star]

            df.to_csv(f"data/anonymized_clean/{filename}", index=False)

print("Done.")