import streamlit as st
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("model.jb")

st.title("Fake News Dectection:-")

st.caption(
    "This model detects linguistic patterns common in real vs fake news datasets. "
    "It does not fact-check statements."
)

st.write("Enter a news article to check if it real or fake")

news_input = st.text_area("News Article:","")
cleaned = clean_text(news_input)

if st.button("Check News"):
    if len(cleaned) == 0:
         st.warning("Please enter a news article to analyze")

    elif len(cleaned.split()) <= 25:
            st.warning("Short texts are less reliable. Please paste a full article.")      

    else:
        transform_input = vectorizer.transform([cleaned])
        prob = model.predict_proba(transform_input)[0]

        if prob[1] >= 0.7:
            st.success("High likelihood of REAL-news style")
        elif prob[1] <= 0.3:
            st.error("High likelihood of FAKE-news style")
        else:
            st.warning("Uncertain — mixed linguistic signals")  
    
     

                