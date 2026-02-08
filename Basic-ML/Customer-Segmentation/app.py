# Customer Segmentation Streamlit App

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Page settings
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Using K-Means Clustering")


# Loading trained model
st.write("Loading trained model...")
model = joblib.load("kmeans_model.pkl")
st.success("Model loaded successfully")


# Loading dataset
st.write("Loading dataset...")
df = pd.read_csv("Mall_Customers.csv")
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
