import pandas as pd
import numpy as np

german_cols = [
    "check_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "install_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "install_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "risk"
]

df_german = pd.read_csv("statlog+german+credit+data/german.data", sep=' ', names=german_cols, skipinitialspace=True)

df_german['risk'] = df_german['risk'].map({1: 0, 2: 1})

df_german.replace(['?', 'unknown'], np.nan, inplace=True)

df_german.to_csv("german_cleaned.csv", index=False, sep=';')