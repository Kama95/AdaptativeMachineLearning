# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from river import tree, naive_bayes, ensemble, drift, metrics
import yfinance as yf

# --- Helper Functions ---
@st.cache_data
def load_financial_data():
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    df = yf.download(tickers, start="2023-01-01", end="2024-12-31")['Close']
    df.dropna(inplace=True)
    df['Target'] = df['AAPL'].diff().apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def stream_batches(df, batch_size=30):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

class AdaptiveModel:
    def __init__(self, model_name='ARF'):
        if model_name == 'ARF':
            self.model = ensemble.AdaptiveRandomForestClassifier()
        elif model_name == 'HT':
            self.model = tree.HoeffdingTreeClassifier()
        elif model_name == 'NB':
            self.model = naive_bayes.GaussianNB()
        else:
            raise ValueError("Invalid model name")
        self.metric = metrics.Accuracy()
        self.kappa = metrics.CohenKappa()
        self.drift_detector = drift.ADWIN()

    def update(self, x, y):
        y_pred = self.model.predict_one(x)
        self.metric.update(y, y_pred)
        self.kappa.update(y, y_pred)
        self.model.learn_one(x, y)
        self.drift_detector.update(int(y_pred != y))
        drift = self.drift_detector.change_detected
        return y_pred, drift

def run_simulation(data, model_name='ARF'):
    model = AdaptiveModel(model_name)
    history = {'accuracy': [], 'kappa': [], 'drift': []}

    for batch in stream_batches(data):
        for _, row in batch.iterrows():
            x = row.drop('Target').to_dict()
            y = row['Target']
            _, drift = model.update(x, y)
            history['drift'].append(drift)
        history['accuracy'].append(model.metric.get())
        history['kappa'].append(model.kappa.get())
    return history

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📈 Adaptive Machine Learning Simulator")

model_choice = st.selectbox("Select Model", ['ARF', 'HT', 'NB'])
if st.button("Run Simulation"):
    with st.spinner("Loading data and running simulation..."):
        df = load_financial_data()
        results = run_simulation(df, model_name=model_choice)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(results['accuracy'], label="Accuracy")
        ax.plot(results['kappa'], label="Cohen's Kappa")
        ax.set_title(f"Model Performance: {model_choice}")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.success("Simulation complete.")
