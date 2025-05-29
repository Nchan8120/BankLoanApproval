import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Model Paths ---
MODEL_PATHS = {
    "Tuned Model": "bankloan_model_tuned.keras",
    "RNN Model": "bankloan_model_rnn.keras",
    "LSTM Model": "bankloan_model_lstm.keras"
}



# Cached model and scaler loading
@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

scaler = load_scaler()

st.title("🏦 Bank Loan Approval Predictor")

# --- Model Selection Dropdown ---
selected_model_name = st.sidebar.selectbox(
    "Select Model:",
    list(MODEL_PATHS.keys())
)
selected_model_path = MODEL_PATHS[selected_model_name]
model = load_keras_model(selected_model_path)
if model is None:
    st.warning("Please ensure all model files and the scaler file exist in the application directory.")
    st.stop() 

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
experience = st.number_input("Work Experience (years)", min_value=0, max_value=80, value=5)
income = st.number_input("Annual Income (in $k)", value=50.0)
family = st.selectbox("Family Size", options=[1, 2, 3, 4])
ccavg = st.number_input("Credit Card Average Spending ($k)", value=1.0)
education = st.selectbox("Education", options=[1, 2, 3], format_func=lambda x: {1: "Highschool", 2: "Undergraduate", 3: "Graduate"}[x])
mortgage = st.number_input("Mortgage Amount ($k)", value=0.0)
securities_account = st.selectbox("Has Securities Account?", options=[0, 1])
cd_account = st.selectbox("Has CD Account?", options=[0, 1])
online = st.selectbox("Uses Online Banking?", options=[0, 1])
creditcard = st.selectbox("Has Credit Card?", options=[0, 1])

# Predict button
if st.button("Predict Loan Approval"):
    # Format data
    input_data = np.array([[age, experience, income, family, ccavg, education, mortgage,
                            securities_account, cd_account, online, creditcard]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]
    result = "✅ Approved" if prediction > 0.5 else "❌ Not Approved"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {prediction:.2%}")
