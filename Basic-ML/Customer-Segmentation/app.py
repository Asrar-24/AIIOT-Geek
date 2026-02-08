# Customer Segmentation Streamlit App

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Page settings
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Using K-Means Clustering")

# Get current folder path
current_dir = os.path.dirname(__file__)

# Full path to model file
model_path = os.path.join(current_dir, "kmeans_model.pkl")

# Loading trained model
st.write("Loading trained model...")
model = joblib.load(model_path)
st.success("Model loaded successfully")

# Full path to CSV file
csv_path = os.path.join(current_dir, "Mall_Customers.csv")

# Loading dataset
st.write("Loading dataset...")
df = pd.read_csv(csv_path)
st.success("Dataset loaded successfully")


# Preview data
st.subheader("Dataset Preview")
st.dataframe(df.head())


# Selecting features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# Predicting clusters
df['Cluster'] = model.predict(X)


# Visualization
st.subheader("Customer Segments Visualization")

fig, ax = plt.subplots()

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    data=df,
    ax=ax
)

ax.set_title("Customer Segments Based on Income and Spending")

st.pyplot(fig)


# Cluster summary
st.subheader("Cluster Summary")

cluster_summary = df.groupby("Cluster")[[
    'Annual Income (k$)',
    'Spending Score (1-100)'
]].mean()

st.dataframe(cluster_summary)


# Insights
st.subheader("Insights")

st.write("""
- Customers are grouped based on income and spending behavior.
- High income and high spending customers form premium segments.
- Low income and low spending customers represent budget segments.
- This segmentation can help in targeted marketing.
""")
