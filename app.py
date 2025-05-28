import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from river import tree, naive_bayes, forest, drift, metrics

# --- Adaptive Model ---
class AdaptiveModel:
    def __init__(self, model_name='ARF'):
        if model_name == 'ARF':
            self.model = forest.ARFClassifier()
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
        drift = self.drift_detector.drift_detected
        return y_pred, drift

# --- Stream Batches ---
def stream_batches(df, batch_size=30):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

# --- Run Simulation ---
def run_simulation(data, model_name='ARF'):
    model = AdaptiveModel(model_name)
    history = {'accuracy': [], 'kappa': [], 'drift': []}
    drift_points = []

    for batch_idx, batch in enumerate(stream_batches(data)):
        for _, row in batch.iterrows():
            x = row.drop('Target').to_dict()
            y = row['Target']
            _, drift = model.update(x, y)
            history['drift'].append(drift)
        history['accuracy'].append(model.metric.get())
        history['kappa'].append(model.kappa.get())
        if any(history['drift'][-len(batch):]):
            drift_points.append(batch_idx)

    return history, drift_points

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📂 Adaptive Machine Learning - CSV Upload Simulator")

uploaded_file = st.file_uploader("Upload your cleaned dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty or 'AAPL_Close' not in df.columns:
            st.error("❌ Dataset must contain the 'AAPL_Close' column to create a target.")
        else:
                        # Step 1: Convert AAPL_Close to numeric and create target
            df['AAPL_Close'] = pd.to_numeric(df['AAPL_Close'], errors='coerce')
            df.dropna(subset=['AAPL_Close'], inplace=True)
            df['Target'] = (df['AAPL_Close'].shift(-1) > df['AAPL_Close']).astype(int)
            df.dropna(inplace=True)

            # Step 2: Drop clearly non-numeric columns (e.g. Date, Text)
            non_numeric = df.select_dtypes(exclude=[np.number]).columns
            df.drop(columns=non_numeric, inplace=True)

            # Step 3: Drop rows with missing values, but only after numeric filtering
            df.dropna(inplace=True)

            # SAFETY CHECK
            if df.shape[0] == 0:
                st.error("❌ No valid numeric rows left after cleaning. Try another dataset or check for empty/missing values.")
                st.stop()

            # Step 4: Separate and scale features
            features = df.drop(columns=['Target'])
            scaler = MinMaxScaler()
            scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
            scaled_features['Target'] = df['Target'].values
            df = scaled_features




            st.success("✅ Dataset loaded and target column generated!")

            # Model selection
            model_choice = st.selectbox("🧠 Choose model to run:", ['ARF', 'HT', 'NB'])

            if st.button("▶ Run Simulation"):
                with st.spinner("Running simulation..."):
                    results, drift_points = run_simulation(df, model_name=model_choice)

                    # Plot Results
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(results['accuracy'], label="Accuracy")
                    ax.plot(results['kappa'], label="Cohen's Kappa")
                    for dp in drift_points:
                        ax.axvline(x=dp, color='red', linestyle='--', alpha=0.5)
                    ax.set_title(f"Model Performance: {model_choice}")
                    ax.set_xlabel("Batch")
                    ax.set_ylabel("Score")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                    # Export Results
                    export_df = pd.DataFrame({
                        "Accuracy": results['accuracy'],
                        "Kappa": results['kappa'],
                        "DriftDetected": [1 if i in drift_points else 0 for i in range(len(results['accuracy']))]
                    })

                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=export_df.to_csv(index=False),
                        file_name=f"{model_choice.lower()}_results.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"❌ Error: {e}")
