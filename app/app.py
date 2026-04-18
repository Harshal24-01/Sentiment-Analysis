import streamlit as st
import joblib
import sys
import os

# Fix import path (IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_sentiment
from keywords import get_top_keywords

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("../model/model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# ---------------------------
# TITLE
# ---------------------------
st.title("🧠 Sentiment Analysis App")
st.write("Analyze tweet sentiment with confidence and insights")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("About")
st.sidebar.info("""
This app uses:
- TF-IDF Vectorization
- Logistic Regression
- NLP preprocessing
""")

# ---------------------------
# USER INPUT
# ---------------------------
user_input = st.text_area("✍️ Enter a Tweet")

# ---------------------------
# ANALYZE BUTTON
# ---------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        prediction, confidence = predict_sentiment(user_input, model, vectorizer)

        # Show result
        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😠 Negative Sentiment")

        # Show confidence text

        # 🔥 ADD THIS HERE
        st.write(f"📊 Confidence Score: **{confidence:.2f}**")
        st.progress(float(confidence))
# ---------------------------
# KEYWORD INSIGHTS
# ---------------------------
st.subheader("🔍 Top Keywords")

top_pos, top_neg = get_top_keywords(model, vectorizer)

col1, col2 = st.columns(2)

with col1:
    st.write("✅ Positive Words")
    for score, word in top_pos:
        st.write(f"{word} ({score:.2f})")

with col2:
    st.write("❌ Negative Words")
    for score, word in top_neg:
        st.write(f"{word} ({score:.2f})")

# ---------------------------
# OPTIONAL: DATA VISUALIZATION
# ---------------------------
st.subheader("📊 Sentiment Distribution")

try:
    import pandas as pd

    df1 = pd.read_csv("../data/data_analysis.csv", low_memory=False)
    df2 = pd.read_csv("../data/data_science.csv", low_memory=False)
    df3 = pd.read_csv("../data/data_visualization.csv", low_memory=False)

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df = df.rename(columns={'tweet': 'text'})

    # recreate sentiment (same logic as training)
    def create_label(text):
        positive_words = ['good', 'great', 'love', 'awesome', 'best', 'amazing', 'happy']
        negative_words = ['bad', 'worst', 'hate', 'poor', 'terrible', 'sad']

        text = str(text).lower()

        if any(word in text for word in positive_words):
            return "positive"
        elif any(word in text for word in negative_words):
            return "negative"
        else:
            return "neutral"

    df['sentiment'] = df['text'].apply(create_label)

    counts = df['sentiment'].value_counts()
    st.bar_chart(counts)

except Exception as e:
    st.warning("⚠️ Could not load dataset for visualization")