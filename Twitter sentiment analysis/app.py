import streamlit as st
import joblib
import re
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
TEXT_COLUMN_ALIASES = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "message",
)

model = joblib.load(APP_DIR / "models" / "lr_model.pkl")
vectorizer = joblib.load(APP_DIR / "models" / "tfidf_vectorizer.pkl")


def find_text_column(columns):
    column_lookup = {
        str(column).strip().lower(): column
        for column in columns
    }
    for alias in TEXT_COLUMN_ALIASES:
        source_column = column_lookup.get(alias)
        if source_column is not None:
            return source_column

    return None

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
st.subheader("Batch CSV Prediction")
uploaded_file = st.file_uploader(
    "Upload a CSV with tweet text",
    type=["csv"],
    help="Supported text columns: text, tweet, tweet_text, full_text, content, message.",
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    text_column = find_text_column(batch_df.columns)

    if text_column is None:
        st.error(
            "CSV must include a text column. Supported names: text, tweet, "
            "tweet_text, full_text, content, message."
        )
    else:
        working_df = batch_df.copy()
        texts = working_df[text_column].fillna("").astype(str)
        cleaned_texts = texts.map(clean_tweet)
        vectors = vectorizer.transform(cleaned_texts)
        predictions = model.predict(vectors)
        probabilities = model.predict_proba(vectors)

        working_df["predicted_sentiment"] = [
            "Positive" if prediction == 1 else "Negative"
            for prediction in predictions
        ]
        working_df["confidence"] = probabilities.max(axis=1)

        st.dataframe(working_df)
        st.download_button(
            "Download Predictions CSV",
            working_df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_predictions.csv",
            mime="text/csv",
        )

st.divider()
st.caption("Model: Logistic Regression | Features: TF-IDF | Dataset: Sentiment140")
