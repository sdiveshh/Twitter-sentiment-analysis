import streamlit as st
import joblib
import re

model = joblib.load(r"C:\Users\ADMIN\OneDrive\Desktop\Twitter sentiment analysis\models\lr_model.pkl")
vectorizer = joblib.load(r"C:\Users\ADMIN\OneDrive\Desktop\Twitter sentiment analysis\models\tfidf_vectorizer.pkl")

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("Predict whether a tweet expresses a Positive or Negative sentiment using a Logistic Regression model trained on the Sentiment140 dataset.")
st.divider()

tweet = st.text_area("✏️ Enter a tweet or any text:", height=120, placeholder="e.g. I love this new feature, it's amazing!")

if st.button("Analyze Sentiment", type="primary"):
    if not tweet.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_tweet(tweet)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        st.divider()
        col1, col2 = st.columns(2)
        if pred == 1:
            col1.success("### 😊 Positive")
        else:
            col1.error("### 😞 Negative")
        col2.metric("Confidence", f"{max(proba):.1%}")
        st.progress(float(proba[1]), text=f"Positive probability: {proba[1]:.1%}")

st.divider()
st.caption("Model: Logistic Regression | Features: TF-IDF | Dataset: Sentiment140")