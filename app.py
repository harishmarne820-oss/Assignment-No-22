import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction")
age = st.number_input("Age")
bp = st.number_input("Resting BP")
cholesterol = st.number_input("Cholesterol")
fasting = st.selectbox("Fasting Blood Sugar", [0, 1])
maxhr = st.number_input("Maximum Heart Rate")
oldpeak = st.number_input("Old Peak")
sex = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    sample = {
        "Age": age,
        "RestingBP": bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting,
        "MaxHR": maxhr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0
    }

    input_df = pd.DataFrame([sample])

    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease : Yes")
    else:
        st.success("Heart Disease : No")