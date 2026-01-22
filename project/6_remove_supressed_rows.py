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

            # Remove rows where age == "*" (indicates full suppression)
            df = df[df["age"].astype(str) != "*"]

            df.to_csv(f"data/anonymized_clean/{filename}", index=False)