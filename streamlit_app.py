import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('best_random_forest_model.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Smart Health Risk Predictor")

# Input Fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", ["Male", "Female"])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# One-hot encoding based on your training data

# cp_1, cp_2, cp_3
cp_val = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
cp_1 = 1 if cp_val == 1 else 0
cp_2 = 1 if cp_val == 2 else 0
cp_3 = 1 if cp_val == 3 else 0

# restecg_1, restecg_2
restecg_val = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}[restecg]
restecg_1 = 1 if restecg_val == 1 else 0
restecg_2 = 1 if restecg_val == 2 else 0

# slope_1, slope_2
slope_val = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
slope_1 = 1 if slope_val == 1 else 0
slope_2 = 1 if slope_val == 2 else 0

# thal_1, thal_2, thal_3
thal_val = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]
thal_1 = 1 if thal_val == 1 else 0
thal_2 = 1 if thal_val == 2 else 0
thal_3 = 1 if thal_val == 3 else 0

# Final input vector with 19 features in exact order
input_data = np.array([[
    age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca,
    cp_1, cp_2, cp_3,
    restecg_1, restecg_2,
    slope_1, slope_2,
    thal_1, thal_2, thal_3
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of heart disease ({prob:.2%} probability)")
    else:
        st.success(f"✅ Low risk of heart disease ({prob:.2%} probability)")
