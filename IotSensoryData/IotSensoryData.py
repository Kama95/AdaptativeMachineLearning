import os
import pandas as pd

# Folder containing all .dat files from the Gas Sensor Array dataset
data_dir = "IotSensoryData/"
combined_data = []

# Loop through all .dat files in the folder
for filename in os.listdir(data_dir):
    if filename.endswith(".dat"):
        file_path = os.path.join(data_dir, filename)
        print(f"Reading {filename}...")
        
        # Read with whitespace delimiter, no headers
        df = pd.read_csv(file_path, sep='\s+', header=None)
        
        # Tag source batch
        df['batch'] = filename
        
        combined_data.append(df)

# Merge all into a single DataFrame
final_df = pd.concat(combined_data, ignore_index=True)

# OPTIONAL: Rename columns (assuming 128 sensors + class + etc.)
# You can change these if you know the correct column layout
num_sensors = final_df.shape[1] - 1  # minus 1 for 'batch'
sensor_columns = [f"sensor_{i+1}" for i in range(num_sensors - 1)] + ["class_label"]  # last column = class
final_df.columns = sensor_columns + ["batch"]

# Save as CSV
final_df.to_csv("gas_sensor_data.csv", index=False)
print("All Gas Sensor Array .dat files processed and saved as gas_sensor_data.csv")
