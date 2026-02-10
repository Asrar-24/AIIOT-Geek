import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Paths
model_path = os.path.join(BASE_DIR, "loan_model.pkl")
feature_path = os.path.join(BASE_DIR, "loan_features.pkl")


# Load objects
model = joblib.load(model_path)
features = joblib.load(feature_path)


st.title("Loan Approval Prediction")
st.write("Check whether loan will be approved")


#  USER INPUTS:

age = st.number_input("Age", 18, 80, 30)
income = st.number_input("Annual Income", 0, 1_000_000, 50000)
emp_exp = st.number_input("Employment Experience (Years)", 0, 50, 5)
credit_score = st.number_input("Credit Score", 300, 900, 650)
loan_amt = st.number_input("Loan Amount", 1000, 1_000_000, 100000)
loan_percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.2)

gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox(
    "Education",
    ["high school", "associate", "bachelor", "master", "doctorate"]
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["rent", "own", "mortgage", "other"]
)

loan_intent = st.selectbox(
    "Loan Intent",
    ["education", "medical", "venture", "personal", "homeimprovement", "debtconsolidation"]
)

previous_default = st.selectbox("Previous Default", ["yes", "no"])


# PREDICT :

if st.button("Check Approval"):

    # Raw input
    input_dict = {
        "person_age": age,
        "person_income": income,
        "person_emp_exp": emp_exp,
        "credit_score": credit_score,
        "loan_amnt": loan_amt,
        "loan_percent_income": loan_percent_income,
        "person_gender": gender,
        "person_education": education,
        "person_home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "previous_loan_defaults_on_file": previous_default
    }

    df_input = pd.DataFrame([input_dict])


    # Encode:
    df_encoded = pd.get_dummies(df_input)


    # Convert bool to int:
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


    # Add missing columns if any:
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0


    # Reorder
    df_encoded = df_encoded[features]


    # Predict
    result = model.predict(df_encoded)[0]


    if result == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
