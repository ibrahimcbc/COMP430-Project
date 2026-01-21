import pandas as pd
import os

os.makedirs("data/clean", exist_ok=True)

# =============================================================================
# ADULT DATASET
# =============================================================================
adult = pd.read_csv("data/raw/adult.csv")

# Drop columns
adult = adult.drop(columns=["fnlwgt", "education-num"])

# Handle missing values - both NaN and "?" strings
for col in ["workclass", "occupation", "native-country"]:
    adult[col] = adult[col].fillna("Unknown")
    adult[col] = adult[col].replace("?", "Unknown")

# Clean target - remove trailing periods
adult["income"] = adult["income"].str.replace(".", "", regex=False)

adult.to_csv("data/clean/adult_clean.csv", index=False)

# =============================================================================
# BANK DATASET
# =============================================================================
bank = pd.read_csv("data/raw/bank.csv")

# Drop columns
drop_cols = ["contact", "day_of_week", "month", "duration",
             "campaign", "pdays", "previous", "poutcome"]
bank = bank.drop(columns=drop_cols)

# Handle missing values - both NaN and "?" strings
for col in ["job", "education"]:
    bank[col] = bank[col].fillna("Unknown")
    bank[col] = bank[col].replace("?", "Unknown")

bank.to_csv("data/clean/bank_clean.csv", index=False)

# =============================================================================
# GERMAN CREDIT DATASET
# =============================================================================
german = pd.read_csv("data/raw/german.csv")

# Rename columns
german.columns = [
    "checking_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status", "other_parties", "residence_since", "property",
    "age", "other_payment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "class"
]

# Clean target - remap 1/2 to good/bad
german["class"] = german["class"].map({1: "good", 2: "bad"})

german.to_csv("data/clean/german_clean.csv", index=False)