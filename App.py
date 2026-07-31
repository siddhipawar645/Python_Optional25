import streamlit as st
import pandas as pd
import joblib


# Load saved files
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


# Title
st.title("Gold Price Prediction")

st.write("Enter the values to predict Gold Price (GLD)")


# Input fields
spx = st.number_input(
    "SPX",
    min_value=0.0,
    value=1400.0
)

uso = st.number_input(
    "USO",
    min_value=0.0,
    value=35.0
)

slv = st.number_input(
    "SLV",
    min_value=0.0,
    value=30.0
)

eur_usd = st.number_input(
    "EUR/USD",
    min_value=0.0,
    value=1.30
)


# Prediction
if st.button("Predict Gold Price"):

    input_data = pd.DataFrame(
        [[spx, uso, slv, eur_usd]],
        columns=["SPX", "USO", "SLV", "EUR/USD"]
    )


    # Scaling
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)


    st.success(
        f"Predicted Gold Price (GLD): {prediction[0]:.2f}"
    )