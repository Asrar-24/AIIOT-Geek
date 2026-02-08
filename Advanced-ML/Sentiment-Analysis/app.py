# Movie Review Sentiment Analysis App:

import joblib
import streamlit as st
import os

# Page settings
st.set_page_config(page_title="Movie Review Sentiment", layout="centered")

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review and check whether it is positive or negative.")

# current folder path on github:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and vectorizer
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")


model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# User's input :
review = st.text_area("Type your movie review here:")

# Prediction button:
if st.button('Predict Sentiment'):
    if review.strip() == "":
        st.warning("Please enter a review first.")
        
    else:
        # Converting text into numeric form unsing the saved vectorizer:
        review_vector = vectorizer.transform([review])

        # Predict sentiment
        result = model.predict(review_vector)

        # result shown:
        if result[0] == 0:
            st.error("Negative Review")

        else:
            st.success("Positive Review")