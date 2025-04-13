# Adaptive Stock Trend Prediction Using River

import pandas as pd
import matplotlib.pyplot as plt
from river import tree, preprocessing, metrics, stream

# STEP 1: Load Cleaned Data
df = pd.read_csv("cleaned_financial_market_data.csv")

# STEP 2: Create Up/Down Label for AAPL
df['PriceChange'] = df['AAPL_Close'].diff().fillna(0)
df['Up'] = (df['PriceChange'] > 0).astype(int)

# STEP 3: Create Monthly Batches for Drift Simulation
df['Date'] = pd.to_datetime(df['Date'])
df['Batch'] = df['Date'].dt.to_period('M')

# STEP 4: Set Features & Target
feature_cols = ['AAPL_Open', 'AAPL_High', 'AAPL_Low', 'AAPL_Volume']
target_col = 'Up'

# STEP 5: Initialize Model & Metric
model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
batch_accuracy = []
batch_labels = []

# STEP 6: Simulate Real-Time Learning by Batch
for batch_name, batch_df in df.groupby('Batch'):
    metric = metrics.Accuracy()
    X_batch = batch_df[feature_cols]
    y_batch = batch_df[target_col]

    for x, y_true in zip(X_batch.to_dict(orient="records"), y_batch):
        y_pred = model.predict_one(x)
        model.learn_one(x, y_true)
        if y_pred is not None:
            metric.update(y_true, y_pred)

    batch_accuracy.append(metric.get())
    batch_labels.append(str(batch_name))

# STEP 7: Plot Accuracy Across Batches
plt.figure(figsize=(10, 5))
plt.plot(batch_labels, batch_accuracy, marker='o', linestyle='-')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Monthly Batches (AAPL)")
plt.grid(True)
plt.tight_layout()
plt.show()
