import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")   # apni model file ka naam yahan likho

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Online Payment Fraud Detection")
st.write("Enter transaction details to check if it's Fraud or Not")

# Input fields (modify according to your dataset features)
step = st.number_input("Step (Time step)", min_value=0)
amount = st.number_input("Amount", min_value=0.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])

# Convert type to numeric (same encoding used during training)
type_dict = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

type_encoded = type_dict[type]

# Prediction button
if st.button("Check Fraud"):
    input_data = np.array([[step, type_encoded, amount, oldbalanceOrg,
                            newbalanceOrig, oldbalanceDest, newbalanceDest]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
