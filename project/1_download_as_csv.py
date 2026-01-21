from ucimlrepo import fetch_ucirepo
import os

os.makedirs("data/raw", exist_ok=True)

# Adult (id=2)
adult = fetch_ucirepo(id=2)
adult_df = adult.data.features.copy()
adult_df["income"] = adult.data.targets
adult_df.to_csv("data/raw/adult.csv", index=False)

# Bank Marketing (id=222)
bank = fetch_ucirepo(id=222)
bank_df = bank.data.features.copy()
bank_df["y"] = bank.data.targets
bank_df.to_csv("data/raw/bank.csv", index=False)

# German Credit (id=144)
german = fetch_ucirepo(id=144)
german_df = german.data.features.copy()
german_df["class"] = german.data.targets
german_df.to_csv("data/raw/german.csv", index=False)