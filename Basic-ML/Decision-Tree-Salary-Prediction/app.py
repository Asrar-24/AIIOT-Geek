import streamlit as st
import joblib
import numpy as np
import os


#current file directory:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Full paths:
model_path = os.path.join(BASE_DIR, "salary_model.pkl")
company_enc_path = os.path.join(BASE_DIR, "company_encoder.pkl")
job_enc_path = os.path.join(BASE_DIR, "job_encoder.pkl")
degree_enc_path = os.path.join(BASE_DIR, "degree_encoder.pkl")


# Loadin objects:
model = joblib.load(model_path)
company_encoder = joblib.load(company_enc_path)
job_encoder = joblib.load(job_enc_path)
degree_encoder = joblib.load(degree_enc_path)


# UI:
st.title("Salary Prediction App: ")
st.write("Predict if salary is more than 100K")


# Inputs
company = st.selectbox("Company", company_encoder.classes_)
job = st.selectbox("Job Role", job_encoder.classes_)
degree = st.selectbox("Degree", degree_encoder.classes_)


if st.button("Predict Salary"):

    # Encode inputs
    company_n = company_encoder.transform([company])[0]
    job_n = job_encoder.transform([job])[0]
    degree_n = degree_encoder.transform([degree])[0]

    X_input = np.array([[company_n, job_n, degree_n]])

    prediction = model.predict(X_input)[0]


    if prediction == 1:
        st.success("✅ Salary is MORE than 100K")
    else:
        st.warning("❌ Salary is LESS than or equal to 100K")
