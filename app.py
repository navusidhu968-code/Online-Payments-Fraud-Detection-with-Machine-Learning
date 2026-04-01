import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Online Payment Fraud Detection")

# -------------------------------
# Load Model Safely
# -------------------------------
model = None

if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
else:
    st.error("❌ model.pkl file not found!")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Transaction Details")

step = st.number_input("Step", min_value=0)
amount = st.number_input("Amount", min_value=0.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

type_option = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

# Encoding
type_dict = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

type_encoded = type_dict[type_option]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check Fraud"):

    if model is None:
        st.warning("⚠️ Model not loaded properly")
    else:
        try:
            input_data = np.array([[step, type_encoded, amount,
                                    oldbalanceOrg, newbalanceOrig,
                                    oldbalanceDest, newbalanceDest]])

            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# -------------------------------
# Debug Section (Optional)
# -------------------------------
st.sidebar.title("🔍 Debug Info")
st.sidebar.write("Files in directory:")
st.sidebar.write(os.listdir())
