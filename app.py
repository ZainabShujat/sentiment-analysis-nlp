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

# Map numeric labels to human-readable form
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Predict sentiment
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Convert numeric prediction to label
        sentiment_label = label_map.get(prediction, "Unknown")
        st.success(f"Predicted Sentiment: **{sentiment_label}**")
