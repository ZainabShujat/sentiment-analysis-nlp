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

# Prediction logic
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]  # 'Positive', 'Neutral', or 'Negative'

        # Show result
        st.success(f"🎯 Predicted Sentiment: **{prediction}**")
