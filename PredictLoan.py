import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and scaler
model = load_model("bankloan_model_tuned.keras")
scaler = joblib.load("scaler.pkl")

# Define feature prompts
features = [
    ("Age", int),
    ("Experience (years)", int),
    ("Annual Income (in $s)", float),
    ("Number of Family Members", int),
    ("Credit Card Average Score (0-10)", float),
    ("Education (1 = High School, 2 = Bachelor's, 3 = Graduate)", int),
    ("Mortgage Amount (in $1000s)", float),
    ("Securities Account (1 = Yes, 0 = No)", int),
    ("CD Account (1 = Yes, 0 = No)", int),
    ("Online Banking (1 = Yes, 0 = No)", int),
    ("Credit Card (1 = Yes, 0 = No)", int)
]

# Collect user input
input_data = []
print("Please enter the following customer details:\n")
for label, dtype in features:
    while True:
        try:
            value = dtype(input(f"{label}: "))
            input_data.append(value)
            break
        except ValueError:
            print(f"Invalid input for {label}. Please enter a {dtype.__name__}.")

# Convert to NumPy and scale
new_data = np.array([input_data])  # shape (1, 11)
new_data_scaled = scaler.transform(new_data)

# Make prediction
prob = model.predict(new_data_scaled)[0][0]
prediction = int(prob >= 0.5)

# Output result
print("\n--- Prediction Result ---")
print(f"Loan Approval Probability: {prob:.4f}")
print("Loan Status:", "✅ APPROVED" if prediction == 1 else "❌ NOT APPROVED")
