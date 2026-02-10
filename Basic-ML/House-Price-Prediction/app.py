import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# ---------------- PATH ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "house_price_model.pkl")
feature_path = os.path.join(BASE_DIR, "model_features.pkl")


# ---------------- LOAD ----------------

model = joblib.load(model_path)
features = joblib.load(feature_path)


# ---------------- UI ----------------

st.set_page_config(page_title="House Price Prediction")

st.title("🏠 House Price Prediction")
st.write("Predict House Price using ML")


# ---------------- INPUTS ----------------

area = st.number_input("Area (sqft)", 500, 10000, 1500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
stories = st.number_input("Stories", 1, 5, 1)
parking = st.number_input("Parking", 0, 5, 1)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)


# ---------------- PREDICT ----------------

if st.button("Predict Price"):

    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,

        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,

        "furnishingstatus": furnishingstatus
    }


    df_input = pd.DataFrame([input_dict])


    # Encode like training
