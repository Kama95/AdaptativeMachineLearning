import pandas as pd

# Load dataset
df = pd.read_csv("internet_usage.csv")

# Quick look at the data
print(df.head())
print(df.info())
