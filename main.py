# main_app.py
import streamlit as st
import os
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.io import fits

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Exoplanet Classification Toolkit 🪐",
    page_icon="🪐",
    layout="wide"
)

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.title("🪐 Exoplanet Classification Toolkit")
mode = st.sidebar.radio(
    "Choose Classifier Mode:",
    (
        "Classify from Tabular Data",
        "Classify from Light Curve (.fits)"
    )
)

# =============================================================================
# MODE 1: TABULAR DATA CLASSIFIER
# =============================================================================
if mode == "Classify from Tabular Data":
    st.title("📊 Exoplanet Candidate Classifier (Tabular Data)")
    st.markdown("""
    Classify **planet candidates** vs **false positives** using trained ML models on tabular data.
    """)

    MODEL_DIR = "models"
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])

    if not model_files:
        st.error("❌ No trained model found in `models/` directory. Please train and save at least one model.")
        st.stop()

    with st.expander("⚙️ Model Selection (click to expand)", expanded=False):
        selected_model_file = st.selectbox(
            "Choose model file:",
            model_files,
            index=0,
            help="Default model is the first one in alphabetical order."
        )

    selected_model_path = os.path.join(MODEL_DIR, selected_model_file)
    best_model = joblib.load(selected_model_path)
    st.success(f"✅ Loaded model: `{selected_model_file}`")

    # --- Model Summary ---
    with st.expander("📊 Model Summary", expanded=False):
        st.write(f"**Model Type:** {type(best_model).__name__}")
        if hasattr(best_model, "feature_names_in_"):
            st.write(f"**Number of Features:** {len(best_model.feature_names_in_)}")
            st.write(f"**Feature Names:** {list(best_model.feature_names_in_)}")
        if hasattr(best_model, "get_params"):
            st.write("**Hyperparameters:**")
            st.json(best_model.get_params())

    # --- Hyperparameter Tuning ---
    with st.expander("🛠️ Hyperparameter Tuning", expanded=False):
        if hasattr(best_model, "get_params") and hasattr(best_model, "set_params"):
            params = best_model.get_params()
            tuned_params = {}
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    tuned_params[key] = st.number_input(f"{key}", value=float(value))
                elif isinstance(value, bool):
                    tuned_params[key] = st.checkbox(f"{key}", value=value)
                elif isinstance(value, str):
                    tuned_params[key] = st.text_input(f"{key}", value=value)
            if st.button("Apply Hyperparameters"):
                best_model.set_params(**tuned_params)
                st.success("✅ Hyperparameters updated!")
        else:
            st.warning("⚠️ This model does not support hyperparameter tuning via this interface.")

    # --- Upload CSV ---
    st.header("📂 Upload Data for Batch Prediction")
    st.info("""
    ### 🧾 CSV Format Instructions
    - Contain **numeric features** used in model training.
    - Identifiers like `tic_id`, `obj_id`, or `star_name` are ignored.
    - Omit the `label` column when predicting new data.
    """)

    uploaded_file = st.file_uploader("📤 Upload a prepared CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        X = data.drop(columns=["label", "tic_id", "obj_id", "object_name", "star_name"], errors="ignore")
        X = X.select_dtypes(include=[float, int])

        if X.empty:
            st.warning("⚠️ No numeric columns found for prediction. Please check your CSV.")
        else:
            preds = best_model.predict(X)
            probs = best_model.predict_proba(X)[:, 1]

            results = data.copy()
            results["prediction"] = preds
            results["probability"] = probs

            st.subheader("🔮 Predictions Preview")
            st.dataframe(results.head(20))

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

    # --- Single Sample Prediction ---
    st.header("🎯 Try Single Prediction")
    sample = {}
    for col in best_model.feature_names_in_:
        sample[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Predict"):
        sample_df = pd.DataFrame([sample])
        pred = best_model.predict(sample_df)[0]
        prob = best_model.predict_proba(sample_df)[0, 1]

        label = "🪐 Planet Candidate" if pred == 1 else "❌ False Positive"
        st.success(f"**Prediction:** {label} (Probability: {prob:.3f})")

# =============================================================================
# MODE 2: LIGHT CURVE (.fits) CLASSIFIER
# =============================================================================
elif mode == "Classify from Light Curve (.fits)":
    st.title("🌌 Exoplanet Candidate Classifier (Light Curve)")
    st.markdown("Upload a `.fits` file to classify it as a potential exoplanet candidate or a false positive.")

    IMAGE_SIZE = 64
    FLUX_COLUMN = 'FLUX'
    TIME_COLUMN = 'TIME'
    MODEL_DIR = "model"
    METRICS_DIR = "metrics"
    MODEL_FILENAME = os.path.join(MODEL_DIR, "cnn_exoplanet_model.h5")

    @st.cache_resource
    def load_keras_model():
        if not os.path.exists(MODEL_FILENAME):
            st.error(f"❌ Model not found: {MODEL_FILENAME}")
            return None
        try:
            model = tf.keras.models.load_model(MODEL_FILENAME)
            return model
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return None

    def process_fits_to_image(file_path, image_size=IMAGE_SIZE):
        try:
            with fits.open(file_path, mode='readonly') as hdul:
                header = hdul[0].header
                data = hdul[1].data
                time = data[TIME_COLUMN]
                flux = data[FLUX_COLUMN]

            lc = lk.LightCurve(time=time, flux=flux).remove_nans()
            if len(lc) == 0:
                return None

            lc_flat = lc.normalize().flatten(window_length=401)
            periodogram = lc_flat.to_periodogram(minimum_period=0.5, maximum_period=30)
            best_period = periodogram.period_at_max_power
            lc_folded = lc_flat.fold(period=best_period)
            binned_lc = lc_folded.bin(bins=image_size)
            flux_values = binned_lc.flux.value

            flux_range = np.max(flux_values) - np.min(flux_values)
            if flux_range == 0:
                image = np.zeros((image_size, image_size))
            else:
                flux_norm = (flux_values - np.min(flux_values)) / flux_range
                if not np.all(np.isfinite(flux_norm)):
                    return None
                image = np.tile(flux_norm, (image_size, 1))

            return {
                "final_image": image,
                "raw_lc": lc,
                "flat_lc": lc_flat,
                "folded_lc": lc_folded,
                "periodogram": periodogram,
                "header": header,
                "best_period": best_period
            }

        except Exception as e:
            st.warning(f"Could not process FITS file: {e}")
            return None

    # --- Model Loading ---
    model = load_keras_model()

    # --- View Performance ---
    with st.expander("📈 View Model Performance Metrics"):
        st.write("Training and evaluation metrics.")
        for metric_file in ["accuracy_curve.png", "loss_curve.png", "confusion_matrix.png", "roc_curve.png"]:
            img_path = os.path.join(METRICS_DIR, metric_file)
            if os.path.exists(img_path):
                st.image(img_path, caption=metric_file)
            else:
                st.warning(f"{metric_file} not found in {METRICS_DIR}/")

    # --- Prediction Interface ---
    if model:
        uploaded_file = st.file_uploader("📤 Upload a .fits file", type=["fits", "fit"])

        with st.expander("⚙️ Advanced Controls"):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.5, 0.01,
                help="Set the confidence required to classify as Exoplanet Candidate."
            )

        if uploaded_file is not None and st.button("Predict", type="primary"):
            with st.spinner("Processing FITS file..."):
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                processed_data = process_fits_to_image(temp_path)
                os.remove(temp_path)

                if processed_data:
                    final_image = processed_data["final_image"]
                    image_reshaped = final_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
                    prediction_prob = model.predict(image_reshaped)[0][0]

                    st.subheader("Prediction Result")
                    if prediction_prob > confidence_threshold:
                        st.success("🪐 **Exoplanet Candidate Detected!**")
                    else:
                        st.error("❌ **False Positive**")

                    st.metric("Model Confidence", f"{prediction_prob:.2%}")

                    fig, ax = plt.subplots()
                    im = ax.imshow(final_image, cmap="viridis", aspect="auto")
                    ax.set_title("Processed Image (Model Input)")
                    st.pyplot(fig)
                else:
                    st.error("Failed to process FITS file.")
