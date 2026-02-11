import streamlit as st
import pandas as pd
import joblib
import os

# Get current directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build file paths
csv_path = os.path.join(BASE_DIR, "sales_prediction.csv")
model_path = os.path.join(BASE_DIR, "sales_model.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")

df = pd.read_csv(csv_path)

# Cache model LOadin:
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns


model, model_columns = load_model()

st.set_page_config(page_title="Sales Forecasting", layout="centered")

st.title("Sales Forecasting System")
st.write("Predict product sales using Machine Learning")

st.header("Enter Product Details")

# Inputs:

item_weight = st.number_input("Item Weight", 0.0, 30.0, 10.0)
item_fat = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_type = st.selectbox("Item Type", df["Item_Type"].dropna().unique().tolist())

item_mrp = st.number_input("Item MRP", 0.0, 300.0, 100.0)

outlet_id = st.selectbox("Outlet Identifier", df["Outlet_Identifier"].dropna().unique().tolist())

outlet_size = st.selectbox("Outlet Size", df["Outlet_Size"].dropna().unique().tolist())

outlet_location = st.selectbox("Outlet Location Type", df["Outlet_Location_Type"].dropna().unique().tolist())

outlet_type = st.selectbox("Outlet Type", df["Outlet_Type"].dropna().unique().tolist())


# Creatin Input DataFrame:

input_dict = {
    "Item_Weight": item_weight,
    "Item_Fat_Content": item_fat,
    "Item_Type": item_type,
    "Item_MRP": item_mrp,
    "Outlet_Identifier": outlet_id,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location,
    "Outlet_Type": outlet_type
}

input_df = pd.DataFrame([input_dict])

# One Hot Encode:

input_encoded = pd.get_dummies(input_df)

# InCase User gives only few columns of info.
input_encoded = input_encoded.reindex(
    columns =  model_columns,
    fill_value = 0
)


# Prediction :

if st.button("Predict Sales"):

    prediction = model.predict(input_encoded)[0]

    st.success(f"Predicted Sales: ₹ {prediction:,.2f}")
