import streamlit as st
import pandas as pd
import joblib

# Title & subtitle
st.title("📝 Sentiment Analysis on Customer Reviews")
st.subheader("Enter a customer review below to predict its sentiment:")

# Input box
user_input = st.text_area("✍️ Review Text")

# Load trained model & vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define label mapping (from LabelEncoder logic)
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Prediction logic
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]  # This will be 0, 1, or 2

        # Convert numeric label to text
        sentiment = label_map.get(prediction, "Unknown")

        st.success(f"🎯 Predicted Sentiment: **{sentiment}**")

