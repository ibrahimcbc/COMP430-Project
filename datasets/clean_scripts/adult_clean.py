import pandas as pd
import numpy as np

cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]

df = pd.read_csv("adult/adult.data", names=cols, skipinitialspace=True)

df.replace('?', np.nan, inplace=True)

print("Missing values per column:")
print(df.isna().sum())

df.to_csv("adult_cleaned.csv", index=False, sep=';')