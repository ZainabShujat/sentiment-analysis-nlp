import streamlit as st
import pandas as pd
import joblib

st.title("📝 Sentiment Analysis on Customer Reviews")
st.subheader("Enter your review below:")

# Input text box
user_input = st.text_area("Review Text")

# Load the trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict sentiment
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
