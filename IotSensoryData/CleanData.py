import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# Load merged gas sensor data
df = pd.read_csv("gas_sensor_data.csv")

# Separate sensor columns and metadata
sensor_columns = [col for col in df.columns if col.startswith("sensor_")]
meta_columns = [col for col in df.columns if col not in sensor_columns]

print(f"Initial rows: {len(df)}")

# 1. Drop rows with any missing values in sensor readings or class label
df.dropna(subset=sensor_columns + ['class_label'], inplace=True)
print(f"After dropping NaNs in sensors and label: {len(df)}")

# 2. Ensure all sensor data is numeric (coerce errors)
df[sensor_columns] = df[sensor_columns].apply(pd.to_numeric, errors='coerce')

# 3. Drop rows with remaining non-numeric entries
df.dropna(subset=sensor_columns, inplace=True)
print(f"After coercing to numeric and dropping NaNs: {len(df)}")

# Stop if DataFrame is empty
if df.empty:
    raise ValueError("No data left after cleaning steps. Please review the dataset.")

# 4. Remove extreme outliers (Z-score based filtering)
df = df[(zscore(df[sensor_columns]) < 3).all(axis=1)]
print(f"After z-score filtering: {len(df)}")

if df.empty:
    raise ValueError("No data left after removing outliers. Consider adjusting the z-score threshold.")

# 5. Normalize sensor values to [0, 1]
scaler = MinMaxScaler()
df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
print(f"After normalization: {len(df)} rows, {len(df.columns)} columns")

# 6. Optional: Encode class labels if they are strings
if df['class_label'].dtype == 'object':
    df['class_label'] = df['class_label'].astype('category').cat.codes
    print("Class labels encoded.")

# Save the cleaned dataset
df.to_csv("cleaned_gas_sensor_data.csv", index=False)
print("✅ Cleaned and preprocessed gas sensor data saved as cleaned_gas_sensor_data.csv")
