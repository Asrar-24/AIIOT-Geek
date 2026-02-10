import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os


# PATH :

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "house_price_model.pkl")
feature_path = os.path.join(BASE_DIR, "model_features.pkl")


#LOAD:

model = joblib.load(model_path)
features = joblib.load(feature_path)


#  UI:

st.set_page_config(page_title="House Price Prediction")

st.title("House Price Prediction")
st.write("Predict house price using Machine Learning")


# INPUTS:

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


# PREDICT:

if st.button("Predict Price"):

    # Raw input to DF:
    df_input = pd.DataFrame([{
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
    }])


    # FEATURE ENGINEERING Columns:-
    df_input["log_area"] = np.log(df_input["area"])

    df_input["room_interaction"] = df_input["bedrooms"] * df_input["bathrooms"]

    df_input["total_rooms"] = df_input["bedrooms"] + df_input["bathrooms"]

    df_input["area_per_room"] = df_input["area"] / df_input["bedrooms"]

    df_input["bath_per_room"] = df_input["bathrooms"] / df_input["bedrooms"]


    # ENCODING :

    df_encoded = pd.get_dummies(df_input)


    # ALIGN MIssing FEATURES if any:

    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[features]


    #  PREDICTION:


    price = model.predict(df_encoded)[0]


    #  OUTPUT :

    st.success(f"Estimated Price: ₹ {int(price):,}")