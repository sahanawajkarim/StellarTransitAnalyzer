# 08_app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- Load best model ---
MODEL_DIR = "../models"
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
best_model_file = os.path.join(MODEL_DIR, model_files[0]) if model_files else None

if best_model_file is None:
    st.error("❌ No trained model found in models/ directory. Please train and save a model first.")
    st.stop()

best_model = joblib.load(best_model_file)
st.sidebar.success(f"Loaded best model: {os.path.basename(best_model_file)}")

# --- App Title ---
st.title("🔭 Exoplanet Candidate Classifier")
st.markdown("Classify **planet candidates** vs **false positives** using the best trained ML model.")

# --- Upload CSV for batch inference ---
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload a prepared CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X = data.drop(columns=["label", "tic_id", "obj_id", "object_name", "star_name"], errors="ignore")
    X = X.select_dtypes(include=[float, int])  # numeric features only
    
    preds = best_model.predict(X)
    probs = best_model.predict_proba(X)[:, 1]
    
    results = data.copy()
    results["prediction"] = preds
    results["probability"] = probs
    
    st.subheader("🔮 Predictions")
    st.write(results.head(20))  # show first 20 rows
    
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# --- Single sample inference ---
st.header("🎯 Try Single Prediction")
st.markdown("Enter feature values manually for prediction:")

# Dynamic input fields
sample = {}
for col in best_model.feature_names_in_:  # available since sklearn 1.0+
    sample[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    sample_df = pd.DataFrame([sample])
    pred = best_model.predict(sample_df)[0]
    prob = best_model.predict_proba(sample_df)[0, 1]
    
    label = "Planet Candidate 🪐" if pred == 1 else "False Positive ❌"
    st.success(f"**Prediction:** {label} (Probability: {prob:.3f})")
