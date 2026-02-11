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

furnishingstatus = st.selectbox( "Furnishing Status",["furnished", "semi-furnished", "unfurnished"] )


# PREDICT:

if st.button("Predict Price"):
    
    # Manual Encoding(Cause One hot was breaking):

    mainroad_yes = 1 if mainroad == "yes" else 0
    guestroom_yes = 1 if guestroom == "yes" else 0
    basement_yes = 1 if basement == "yes" else 0
    hotwaterheating_yes = 1 if hotwaterheating == "yes" else 0
    airconditioning_yes = 1 if airconditioning == "yes" else 0
    prefarea_yes = 1 if prefarea == "yes" else 0

    semi_furnished = 1 if furnishingstatus == "semi-furnished" else 0
    unfurnished = 1 if furnishingstatus == "unfurnished" else 0

    # Feature Engineering (Asin Notebook):

    area_per_room = area / bedrooms
    bath_per_room = bathrooms / bedrooms
    total_rooms = bedrooms + bathrooms
    log_area = np.log1p(area)

    # Building Final Input: 

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad_yes": mainroad_yes,
        "guestroom_yes": guestroom_yes,
        "basement_yes": basement_yes,
        "hotwaterheating_yes": hotwaterheating_yes,
        "airconditioning_yes": airconditioning_yes,
        "prefarea_yes": prefarea_yes,
        "furnishingstatus_semi-furnished": semi_furnished,
        "furnishingstatus_unfurnished": unfurnished,
        "area_per_room": area_per_room,
        "bath_per_room": bath_per_room,
        "total_rooms": total_rooms,
        "log_area": log_area
    }


    df_encoded = pd.DataFrame([input_data])

    # To Ensure correct order
    df_encoded = df_encoded[features]
  
    #  PREDICTION:

    log_price = model.predict(df_encoded)[0]

    price = np.expm1(log_price)


    #  OUTPUT :

    st.success(f"Estimated Price: ₹ {int(price):,}")