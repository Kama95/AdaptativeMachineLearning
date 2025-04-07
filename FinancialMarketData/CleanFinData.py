import pandas as pd

# Load raw file
raw_df = pd.read_csv("financial_market_data.csv", skiprows=1)

# Use the second row as header
new_header = raw_df.iloc[0]
df = raw_df[1:].copy()
df.columns = new_header

# Drop rows with any missing date or price info
df = df.dropna(subset=['Date'])

# Convert all applicable columns to numeric
for col in df.columns:
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Final cleanup: drop rows with NaN in any important columns
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

# Save cleaned file
df.to_csv("cleaned_financial_market_data.csv", index=False)

print("✅ Data cleaned and saved as 'cleaned_financial_market_data.csv'")
