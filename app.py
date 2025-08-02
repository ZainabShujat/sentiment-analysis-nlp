import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained model & vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Label mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- 🌟 UI SETUP ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="🛍️", layout="centered")

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #FF4B4B;'>🛍️ Sentiment Analysis on Customer Reviews</h1>
        <p style='font-size: 18px;'>Using Naive Bayes and NLP to understand what your customers feel</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Optional hero image or banner
st.image("https://img.freepik.com/free-photo/fashionable-woman-shopping-bags_23-2147775305.jpg", use_column_width=True)

# Text Input
st.markdown("### ✍️ Enter a customer review:")
user_input = st.text_area("Your Review", placeholder="e.g. Loved the fabric and fit, would definitely recommend!")

# Prediction
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        sentiment = label_map.get(prediction, "Unknown")

        # Result display with emojis
        st.markdown("---")
        st.markdown(f"## 🎯 Predicted Sentiment: **:blue[{sentiment}]**")

        if sentiment == "Positive":
            st.success("Customers are loving it! 💖")
        elif sentiment == "Negative":
            st.error("Oops! Might be time for a product review 👀")
        elif sentiment == "Neutral":
            st.info("It's okay... not bad, not great 🤷‍♀️")
