import pandas as pd

df = pd.read_parquet("data/cache/features_full.parquet")
print(df["win"].value_counts())
