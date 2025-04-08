import pandas as pd

# Load the dataset
df = pd.read_csv("internet_usage.csv")

# Replace '..' with NaN
df.replace("..", pd.NA, inplace=True)

# Convert year columns to numeric
year_columns = df.columns[2:]  # from 2000 to 2023
df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows where all year values are missing
df.dropna(subset=year_columns, how='all', inplace=True)

# Forward-fill missing values along each row (per country)
df[year_columns] = df[year_columns].ffill(axis=1)

# Save cleaned dataset
df.to_csv("cleaned_internet_usage.csv", index=False)
print("✅ Cleaned dataset saved as cleaned_internet_usage.csv")
