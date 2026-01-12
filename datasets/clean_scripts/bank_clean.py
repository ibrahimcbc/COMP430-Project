import pandas as pd
import numpy as np

df_bank = pd.read_csv("bank+marketing/bank-additional-full.csv", sep=';')

df_bank = df_bank.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

df_bank.replace('unknown', np.nan, inplace=True)

df_bank.to_csv("bank_cleaned.csv", index=False, sep=';')