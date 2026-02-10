import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# Base directory:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Model & Feature Paths:
model_path = os.path.join(BASE_DIR, "house_price_model.pkl")
feature_path = os.path.join(BASE_DIR, "model_features.pkl")


# Load
model = joblib.load(model_path)
features = joblib.load(feature_path)


st.title("House Price Prediction")
st.write("Predict House Price using ML")


# USER INPUTS:

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
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)


# PREDICTION Part:

if st.button("Predict Price"):

# Create raw Dict input to convert into DF later:
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
        "furnishingstatus": furnishingstatus
    }

    df_input = pd.DataFrame([input_dict])

    # Encode like training
    df_encoded = pd.get_dummies(df_input, drop_first=True)

    # Add missing columns as 0 if missing.
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns
    df_encoded = df_encoded[features]

    # Predict
    price = model.predict(df_encoded)[0]

    st.success(f"Estimated Price: ₹ {int(price):,}")
