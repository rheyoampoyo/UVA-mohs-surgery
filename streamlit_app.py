import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# Load models and artifacts
# ==========================

base_dir = "Prod/models"

classifier = joblib.load(os.path.join(base_dir, "xgb_classifier_visit_complexity_kmeans.pkl"))
regressor = joblib.load(os.path.join(base_dir, "xgb_regressor_visit_duration.pkl"))
kmeans_regressor = joblib.load(os.path.join(base_dir, "xgb_regressor_normalized_visit_complexity_kmeans.pkl"))

feature_names = joblib.load(os.path.join(base_dir, "model1_features.pkl"))
label_encoder = joblib.load(os.path.join(base_dir, "label_encoder_visit_complexity_kmeans.pkl"))

# ==========================
# UI setup
# ==========================

st.set_page_config(page_title="Visit Complexity & Duration Estimator", layout="centered")
st.title("🧠 Patient Visit Complexity Score & Duration Estimator")
st.markdown("Enter patient details to estimate their **complexity category**, **normalized KMeans complexity score**, and **expected duration of visit**.")

# ==========================
# Sidebar Inputs
# ==========================

st.sidebar.header("Input Patient Features")

recurrent_tumor = st.sidebar.selectbox("Recurrent Tumor (Y/N)", ["yes", "no"])
aggressive_histology = st.sidebar.selectbox("Aggressive Histology (Y/N)", ["yes", "no"])
wound_management = st.sidebar.selectbox("Wound Management (H/M/L)", ["H", "M", "L"])
location = st.sidebar.selectbox("Location (H/M/L)", ["H", "M", "L"])
immunosuppressed = st.sidebar.selectbox("Immunosuppressed (Y/N)", ["yes", "no"])
bleeding_risk = st.sidebar.selectbox("Bleeding Risk (Y/N)", ["yes", "no"])
greater_avg_time = st.sidebar.selectbox("Greater Average Time (Y/N)", ["yes", "no"])

age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50)
lesion_size = st.sidebar.number_input("Lesion Size (cm)", min_value=0.0, value=1.0)
treatment_delay = st.sidebar.number_input("Treatment Delay (days)", min_value=0, value=0)

if st.sidebar.button("Estimate"):

    # ==========================
    # Input dict
    # ==========================
    input_dict = {
        "Age (years)": age,
        "Lesion Size (cm)": lesion_size,
        "Treatment Delay (days)": treatment_delay,
        "Recurrent Tumor (Y/N)": recurrent_tumor,
        "Aggressive Histology (Y/N)": aggressive_histology,
        "Wound Management (H/M/L)": wound_management,
        "Location (H/M/L)": location,
        "Immunosuppressed (Y/N)": immunosuppressed,
        "Bleeding Risk (Y/N)": bleeding_risk,
        "Greater Average Time (Y/N)": greater_avg_time
    }

    input_df = pd.DataFrame([input_dict])

    # ==========================
    # Fix column name typo globally
    # ==========================
    input_df = input_df.rename(columns={'Lesion Size (cm)': 'Lesion  Size (cm)'})  # Match training double-space

    # ==========================
    # Classifier path (one-hot encoded)
    input_encoded_clf = pd.get_dummies(input_df)

    # Ensure consistent column names and ordering
    input_encoded_clf.columns = input_encoded_clf.columns.str.strip().str.replace("  ", " ", regex=False)

    for col in feature_names:
        if col not in input_encoded_clf.columns:
            input_encoded_clf[col] = 0

    input_encoded_clf = input_encoded_clf[feature_names]

    probas = classifier.predict_proba(input_encoded_clf)[0]
    pred_class_idx = np.argmax(probas)
    pred_class_label = label_encoder.inverse_transform([pred_class_idx])[0]
    pred_class_confidence = probas[pred_class_idx]

    # ==========================
    # Regressor path (native categorical)
    regressor_features = [
        'Lesion  Size (cm)',  # Double space preserved
        'Recurrent Tumor (Y/N)',
        'Aggressive Histology (Y/N)',
        'Wound Management (H/M/L)',
        'Location (H/M/L)',
        'Treatment Delay (days)',
        'Age (years)',
        'Immunosuppressed (Y/N)',
        'Bleeding Risk (Y/N)',
        'Greater Average Time (Y/N)'
    ]

    cat_cols = [
        'Recurrent Tumor (Y/N)',
        'Aggressive Histology (Y/N)',
        'Wound Management (H/M/L)',
        'Location (H/M/L)',
        'Immunosuppressed (Y/N)',
        'Bleeding Risk (Y/N)',
        'Greater Average Time (Y/N)'
    ]

    input_reg = input_df.copy()

    for col in cat_cols:
        input_reg[col] = input_reg[col].astype('category')

    input_reg = input_reg[regressor_features]

    predicted_duration = regressor.predict(input_reg)[0]
    predicted_kmeans_score = kmeans_regressor.predict(input_reg)[0]

    # ==========================
    # Output results
    st.subheader("🩺 Estimation Result")
    st.info(f"Estimated Complexity Score: **{predicted_kmeans_score * 100:.2f}**")
    st.info(f"Estimated Visit Duration: **{predicted_duration:.1f} minutes**")