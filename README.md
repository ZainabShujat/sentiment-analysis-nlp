# 🧠 Sentiment Analysis of Customer Reviews using NLP & Naive Bayes

This project analyzes customer review texts to determine whether the sentiment expressed is **positive or negative**, using **Natural Language Processing (NLP)** and a **Naive Bayes classifier**. It applies fundamental text preprocessing techniques and trains a model to classify sentiment accurately.
---
Live at : https://zainab-sentiment-analysis.streamlit.app/ 
---

## 🔍 Objective

Build an end-to-end sentiment classification pipeline using NLP preprocessing techniques and the Naive Bayes algorithm. Evaluate the model using standard metrics and visualize insights from the dataset.

---

## ⚙️ Technologies & Libraries Used

- **Python** (3.x)
- **Jupyter Notebook**
- **NLTK** (for stopwords, tokenization)
- **Scikit-learn** (for model training, TF-IDF, evaluation)
- **Pandas & NumPy** (for data handling)
- **Matplotlib & Seaborn** (for visualizations)
- **WordCloud & Pillow** (for word cloud analysis)

---

## 🗂️ Dataset

A CSV file of customer reviews containing text data and sentiment labels (`positive` or `negative`). Publicly available datasets such as those from **Kaggle** or similar platforms can be used.

---

## 📊 Features & Output

The notebook performs the following:

- ✅ Text Preprocessing: tokenization, lowercasing, stopword removal, stemming
- ✅ Feature Extraction using **TF-IDF Vectorization**
- ✅ Sentiment Classification using **Multinomial Naive Bayes**
- ✅ Model Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix
- ✅ Word Cloud Visualization for both positive and negative sentiments

---

## 📌 Notes

- This project was completed as part of an individual internship task under **IBM SkillBuild**.
- The notebook is self-contained and runs end-to-end.
- Final submission will be made via GitHub.

---

## 🚀 Future Enhancements (Optional)

- Add neutral sentiment class for multi-class classification
- Try alternate models (e.g., SVM, Logistic Regression)
- Experiment with spaCy for faster preprocessing
