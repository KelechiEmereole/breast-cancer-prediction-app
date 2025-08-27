import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model
try:
    loaded_model = joblib.load("logistic_regression_model.sav")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    loaded_model = None

# Load Scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Scaler file not found. Make sure 'scaler.pkl' is in the correct location.")
    scaler = None


def main():
    st.title("🩺 Breast Cancer Prediction App")

    st.write("Enter the feature values below to predict whether the tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**.")

    # Create input fields for the selected 13 features
    col1, col2 = st.columns(2)

    with col1:
        texture_worst = st.number_input("Texture Worst", min_value=0.0, step=0.1)
        radius_se = st.number_input("Radius SE", min_value=0.0, step=0.0001, format="%.4f")
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, step=0.0001, format="%.4f")
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, step=0.00001, format="%.5f")
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, step=0.0001, format="%.4f")
        area_se = st.number_input("Area SE", min_value=0.0, step=0.1)
        compactness_se = st.number_input("Compactness SE", min_value=0.0, step=0.00001, format="%.5f")

    with col2:
        area_worst = st.number_input("Area Worst", min_value=0.0, step=1.0)
        radius_worst = st.number_input("Radius Worst", min_value=0.0, step=0.001, format="%.3f")
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, step=0.00001, format="%.5f")
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0, step=0.001, format="%.3f")
        perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, step=1.0)
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, step=0.00001, format="%.5f")

    # Predict Button
    if st.button("🔍 Predict", use_container_width=True):
        if loaded_model is not None and scaler is not None:
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    "texture_worst": [texture_worst],
                    "radius_se": [radius_se],
                    "symmetry_worst": [symmetry_worst],
                    "concave_points_mean": [concave_points_mean],
                    "concavity_worst": [concavity_worst],
                    "area_se": [area_se],
                    "compactness_se": [compactness_se],
                    "area_worst": [area_worst],
                    "radius_worst": [radius_worst],
                    "concavity_mean": [concavity_mean],
                    "perimeter_se": [perimeter_se],
                    "perimeter_worst": [perimeter_worst],
                    "concave_points_worst": [concave_points_worst]
                })

                # Scale input
                input_scaled = scaler.transform(input_data)

                # Make prediction
                prediction = loaded_model.predict(input_scaled)
                prediction_proba = loaded_model.predict_proba(input_scaled)[0][1]

                if prediction[0] == 1:
                    st.error(f"⚠️ Prediction: Malignant (Cancerous) \n\n Probability: {prediction_proba:.2%}")
                else:
                    st.success(f"Prediction: Benign (Non-Cancerous) \n\n Probability: {prediction_proba:.2%}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("❌ Model or Scaler not loaded. Please check the files.")


if __name__ == "__main__":
    main()
