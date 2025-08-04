import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Customer Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# --- LOAD MODEL + VECTORIZER ---
try:
    model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
emoji_map = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}

# --- HEADER ---
st.title("🛒 Amazon Product Review Sentiment Analyzer")
st.markdown("""
Welcome to the **Sentiment Analysis App**!

This tool helps you understand how customers feel about a product based on their review.

Enter any review below — it could be glowing, angry, or just *meh* — and we'll tell you what it sounds like.
""")

# --- TABS ---
tab1, tab2 = st.tabs(["🔍 Predict Sentiment", "📊 Dataset Insights"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.subheader("📋 Write or paste a customer review:")
    user_input = st.text_area("✍️ Review Text", height=150, placeholder="e.g. The fabric was soft and the fit was perfect!")

    if st.button("🔍 Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Oops! Please enter a review before analyzing.")
        else:
            try:
                transformed_input = vectorizer.transform([user_input])
                prediction = model.predict(transformed_input)[0]
                sentiment = label_map.get(prediction, "Unknown")
                emoji = emoji_map.get(sentiment, "❓")

                st.markdown(f"""
                ## 🎯 Sentiment Detected: **{sentiment}** {emoji}
                """)
                st.info("Note: This prediction is based on past customer reviews. Real-world interpretation may vary.")
            except Exception as e:
                st.error(f"Error analyzing sentiment: {e}")

# --- TAB 2: INSIGHTS ---
with tab2:
    st.title("📊 Visual Insights from the Dataset")
    st.markdown("These WordClouds show the most frequent words in each sentiment category:")

    col1, col2, col3 = st.columns(3)

    for col, sentiment, emoji in zip([col1, col2, col3], ["positive", "negative", "neutral"], ["🌟", "💢", "😐"]):
        image_path = f"assets/wordcloud_{sentiment}.png"
        with col:
            if os.path.exists(image_path):
                st.image(image_path, caption=f"{emoji} {sentiment.capitalize()} Reviews", use_container_width=True)
            else:
                st.warning(f"{sentiment.capitalize()} wordcloud image not found")

    st.markdown("---")
    st.subheader("🎯 Confusion Matrix Heatmap")
    st.markdown("This heatmap shows how well the model performed across each sentiment class.")

    heatmap_path = "assets/confusion_matrix.png"
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, use_container_width=True)
    else:
        st.warning("Confusion matrix image not found")

# --- FOOTER ---
st.markdown("""
<hr style='border:1px solid #f0f0f0'>
<small>Project by Zainab Shujat 💛 | Powered by Streamlit + Naive Bayes</small>
""", unsafe_allow_html=True)
