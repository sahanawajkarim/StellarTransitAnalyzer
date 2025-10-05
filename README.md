# 🪐 Stellar Transit Analyzer

The **Stellar Transit Analyzer** is an integrated platform that uses **Machine Learning (ML)** and **Deep Learning (DL)** to automate the classification of **exoplanet candidates** versus **false positives** using both **tabular astrophysical data** and **light curve (.fits) data**. The project combines exploratory research, model experimentation, and deployment into a single modular workflow accessible via a unified **Streamlit application**.

---

## 🚀 Project Overview

Exoplanet detection is a challenging task due to noisy signals, false transit detections, and correlated astrophysical features. This toolkit addresses these challenges by:

* Using **ensemble ML models** (Random Forest, LightGBM, CatBoost, XGBoost) on tabular astrophysical features.
* Leveraging a **Convolutional Neural Network (CNN)** for raw light curve classification (.fits files).
* Providing an easy-to-use **Streamlit web app** with multiple modes:

  * **Classify from Tabular Data**
  * **Classify from Light Curve (.fits)**
  * **Experiments with the Tabular Data and ML Models**
  * **Experiments Related to Lightcurve Data**

The system enables both **batch inference** and **single prediction**, along with visualization of experiments, metrics, and plots.

---

## 📂 Repository Structure

```
.
├── models/                                # Saved ML models (.pkl)
├── model/                                 # CNN model for light curves (.h5)
├── metrics/                               # Model metrics, ROC curves, confusion matrices
├── plots_of_experiment_on_tabular_dataset/ # Experiment plots for tabular models
│   ├── baseline_results_plot.png
│   ├── ensemble_results_plot.png
│   └── feature_correlation_heatmap.png
├── Plots_Related_to_lightcurve_data/      # Experiment plots for light curve models
│   ├── best_threshold.png
│   ├── metric_correlation_heatmap.png
│   └── model_performance_comparison.png
├── notebooks/                             # Jupyter notebooks for modular experimentation
│   ├── 01_EDA_and_data_checks.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_ensembles_exploration.ipynb
│   ├── 04_hyperparam_tuning.ipynb
│   ├── 05_stacking_and_meta.ipynb
│   ├── 06_model_interpretation.ipynb
│   └── 07_final_models_and_inference.ipynb
├── main_app.py                            # Unified Streamlit application
└── README.md                              # Project documentation
```

---

## 🧪 Features

### **1. Tabular Data Classifier**

* Supports batch CSV uploads and single prediction inputs.
* Dynamic model selection from `models/` folder.
* Experiment visualization for baseline, ensembles, and correlations.

### **2. Light Curve Classifier**

* Preprocessing pipeline with **Lightkurve**:

  * Normalization, folding, binning.
* CNN-based classifier trained on processed `.fits` curves.
* Supports prediction, confidence scores, and visualization.

### **3. Experiment Visualization**

* **Tabular Experiments:** Baselines, ensemble results, and feature correlations.
* **Light Curve Experiments:** Best thresholds, metric correlation heatmaps, and model performance comparisons.

### **4. Deployment-Ready Streamlit App**

* Unified app with sidebar navigation for all modes.
* Expandable panels for model selection, summaries, and hyperparameter tuning.
* Interactive plots and performance metrics.

---

## ⚙️ Tech Stack

**Languages & Frameworks**

* Python 3
* Scikit-learn, TensorFlow/Keras
* Lightkurve, Astropy
* Pandas, NumPy, Matplotlib
* Optuna (hyperparameter tuning)
* SHAP (interpretability)
* Streamlit (web deployment)

**Supporting Tools**

* Joblib (model serialization)
* Git & GitHub (version control)
* Jupyter Notebooks (experimentation)

---

## 📊 Workflow

The project follows a modular workflow with dedicated notebooks:

1. **01_EDA_and_data_checks.ipynb** – EDA, sanity checks, correlations.
2. **02_baselines.ipynb** – Quick baseline models (LR, DT, KNN, NB).
3. **03_ensembles_exploration.ipynb** – Ensembles (RF, XGB, LGBM, CatBoost).
4. **04_hyperparam_tuning.ipynb** – RandomizedSearch, GridSearch, Optuna.
5. **05_stacking_and_meta.ipynb** – Stacking meta-models and ensembles.
6. **06_model_interpretation.ipynb** – SHAP, feature importance, PDPs.
7. **07_final_models_and_inference.ipynb** – Export and inference pipeline.

---

## 🌍 Benefits & Impact

* **Automation**: Replaces manual screening of exoplanet candidates with automated pipelines.
* **Accuracy**: Improves classification reliability via ensembles and CNNs.
* **Interpretability**: Provides SHAP plots, feature importances, calibration curves.
* **Scalability**: Flexible, modular design supports integration with new missions.
* **Accessibility**: Single unified Streamlit app for researchers and practitioners.

---

## 🖥️ Running the App

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Exoplanet-Classification-Toolkit.git
cd Exoplanet-Classification-Toolkit
```


3. Run the Streamlit app:

```bash
streamlit run main_app.py
```

4. Navigate to the app in your browser (default: `http://localhost:8501`).

---

## 🎨 Creativity & Design Considerations

* **Dual-mode classification** (tabular + light curve) in one toolkit.
* **Unified UI** with sidebar-based navigation.
* **Experiment visualization integration** for research transparency.
* **Reproducibility** ensured with modular notebooks and logged hyperparameters.

---

## 📌 Future Work

* Integration with **NASA Exoplanet Archive API** for live data updates.
* Incorporation of **transfer learning** for CNN improvements.
* Support for **multi-class classification** (beyond candidate/false positive).
* Expansion with **explainable AI dashboards**.

---

## 🙌 Acknowledgements

* **NASA Exoplanet Archive** for providing datasets.
* **Lightkurve** and **Astropy** teams for open-source libraries.
* OpenAI (AI assistance in coding, debugging, and research brainstorming).

---

## 📜 License

This project is licensed under the MIT License.
