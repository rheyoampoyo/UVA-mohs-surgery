import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# Load models and expected feature list
# ==========================
base_dir = "Prod/models"

# Load models
duration_regressor = joblib.load(os.path.join(base_dir, "xgb_regressor_visit_duration.pkl"))
stages_regressor = joblib.load(os.path.join(base_dir, "xgb_regressor_number_of_stages.pkl"))
anesthetic_regressor = joblib.load(os.path.join(base_dir, "xgb_regressor_anesthetic_amount.pkl"))

feature_columns = joblib.load(os.path.join(base_dir, "model1_features.pkl"))

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Mohs Surgery Outcome Estimator", layout="centered")
st.title("🧠 Mohs Surgery Outcome Estimator")
st.markdown("Enter patient details to estimate expected **visit duration**, **number of stages**, and **anesthetic amount**.")

# ==========================
# Sidebar Inputs
# ==========================
st.sidebar.header("Input Patient Features")

recurrent_tumor = st.sidebar.selectbox("Recurrent Tumor", ["yes", "no"])
aggressive_histology = st.sidebar.selectbox("Aggressive Histology", ["yes", "no"])
wound_management = st.sidebar.selectbox("Wound Management", ["H", "M", "L"])
location = st.sidebar.selectbox("Location", ["H", "M", "L"])
treatment_delay = st.sidebar.number_input("Treatment Delay (days)", min_value=0, value=0)
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=60)
lesion_size = st.sidebar.number_input("Lesion Size (cm)", min_value=0.0, value=1.0)
immunosuppressed = st.sidebar.selectbox("Immunosuppressed", ["yes", "no"])
bleeding_risk = st.sidebar.selectbox("Bleeding Risk", ["yes", "no"])
greater_avg_time = st.sidebar.selectbox("Greater Average Time", ["yes", "no"])

# ==========================
# Predict Button
# ==========================
if st.sidebar.button("Estimate"):
    input_dict = {
        "Lesion  Size (cm)": lesion_size,
        "Recurrent Tumor (Y/N)": recurrent_tumor,
        "Aggressive Histology (Y/N)": aggressive_histology,
        "Wound Management (H/M/L)": wound_management,
        "Location (H/M/L)": location,
        "Treatment Delay (days)": treatment_delay,
        "Age (years)": age,
        "Immunosuppressed (Y/N)": immunosuppressed,
        "Bleeding Risk (Y/N)": bleeding_risk,
        "Greater Average Time (Y/N)": greater_avg_time
    }

    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    duration_preds_all = duration_regressor.predict(input_encoded, iteration_range=(0, duration_regressor.n_estimators))
    stages_preds_all = stages_regressor.predict(input_encoded)
    anesthetic_preds_all = anesthetic_regressor.predict(input_encoded, iteration_range=(0, anesthetic_regressor.n_estimators))

    duration_mean = np.mean(duration_preds_all)

    stages_mean = np.mean(stages_preds_all)

    anesthetic_mean = np.mean(anesthetic_preds_all)

# ==========================
# Output Results
# ==========================
    st.subheader("📋 Prediction Results")

    st.success(
        f"**Estimated Visit Duration:** {duration_mean:.1f} minutes "
    )

    st.success(
        f"**Estimated Number of Stages:** {stages_mean:.1f} "
    )

    st.success(
        f"**Estimated Anesthetic Amount:** {anesthetic_mean:.1f} ml "
    )