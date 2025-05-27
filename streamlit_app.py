import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Cached model and scaler loading
@st.cache_resource
def load_keras_model():
    return load_model("bankloan_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_keras_model()
scaler = load_scaler()

st.title("🏦 Bank Loan Approval Predictor")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
experience = st.number_input("Work Experience (years)", min_value=0, max_value=80, value=5)
income = st.number_input("Annual Income (in $k)", value=50.0)
family = st.selectbox("Family Size", options=[1, 2, 3, 4])
ccavg = st.number_input("Credit Card Average Spending ($k)", value=1.0)
education = st.selectbox("Education", options=[1, 2, 3], format_func=lambda x: {1: "Highschool", 2: "Undergraduate", 3: "Graduate"}[x])
mortgage = st.number_input("Mortgage Amount ($k)", value=0.0)
securities_account = st.selectbox("Has Securities Account?", options=["Yes", "No"])
cd_account = st.selectbox("Has CD Account?", options=["Yes", "No"])
online = st.selectbox("Uses Online Banking?", options=["Yes", "No"])
creditcard = st.selectbox("Has Credit Card?", options=["Yes", "No"])

# Predict button
if st.button("Predict Loan Approval"):
    # Convert Yes/No to 1/0
    securities_account_val = 1 if securities_account == "Yes" else 0
    cd_account_val = 1 if cd_account == "Yes" else 0
    online_val = 1 if online == "Yes" else 0
    creditcard_val = 1 if creditcard == "Yes" else 0

    # Format data
    input_data = np.array([[age, experience, income, family, ccavg, education, mortgage,
                            securities_account_val, cd_account_val, online_val, creditcard_val]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]
    result = "✅ Approved" if prediction > 0.5 else "❌ Not Approved"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {prediction:.2%}")
