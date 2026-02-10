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


if st.button("Predict Price"):

    try:

        st.subheader("🔍 DEBUG MODE")

        # Raw input
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

        st.write("1️⃣ Raw Input Dict:")
        st.json(input_dict)


        # DataFrame
        df_input = pd.DataFrame([input_dict])

        st.write("2️⃣ Input DataFrame:")
        st.dataframe(df_input)


        # Dummies
        df_encoded = pd.get_dummies(df_input, drop_first=True)

        st.write("3️⃣ After get_dummies():")
        st.dataframe(df_encoded)


        # Missing columns
        missing = []
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                missing.append(col)

        st.write("4️⃣ Missing Columns Added:")
        st.write(missing)


        # Extra columns
        extra = [c for c in df_encoded.columns if c not in features]

        st.write("5️⃣ Extra Columns Removed:")
        st.write(extra)


        # Align
        df_encoded = df_encoded[features]

        st.write("6️⃣ Final Model Input:")
        st.dataframe(df_encoded)


        # Prediction
        log_price = model.predict(df_encoded)[0]

        st.write("7️⃣ Raw Model Output (log):", log_price)


        # Reverse
        price = np.expm1(log_price)

        st.write("8️⃣ Final Price:", price)


        st.success(f"💰 Estimated Price: ₹ {int(price):,}")


    except Exception as e:

        st.error("❌ ERROR")
        st.exception(e)

