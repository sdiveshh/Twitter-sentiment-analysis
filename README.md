# Twitter Sentiment Analysis

An end-to-end machine learning project that classifies tweets as **Positive** or **Negative**,
built with Python, Pandas, Scikit-learn, and Streamlit.

## About the Dataset

The dataset (`tweets.csv`) is the **Sentiment140** dataset containing 1.6 million tweets,
labelled 0 (Negative) or 4 (Positive). Downloaded from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Project Workflow

1. **Data Cleaning** — Loaded the dataset, remapped labels (0/1), removed nulls, stripped URLs/mentions/special characters.
2. **Exploratory Data Analysis (EDA)** — Visualized sentiment distribution and tweet length patterns by class.
3. **Feature Engineering** — Applied TF-IDF vectorization (50k features, unigrams + bigrams) to convert text to numerical format.
4. **Model Training** — Trained a Logistic Regression classifier.
5. **Evaluation** — Achieved ~79% accuracy; analyzed confusion matrix and classification report.
6. **Deployment** — Built an interactive Streamlit app for real-time sentiment prediction.

## Tools Used

- Python
- Pandas
- Seaborn & Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## How to Run

```bash
pip install pandas scikit-learn streamlit joblib seaborn matplotlib
streamlit run "Twitter sentiment analysis/app.py"
```

## Batch CSV Prediction

The Streamlit app can analyze uploaded CSV files. Use any of these text column
names: `text`, `tweet`, `tweet_text`, `full_text`, `content`, or `message`.

Reviewed exports from tools such as TweetClaw can be saved as CSV with one of
those columns, then uploaded to generate `predicted_sentiment` and `confidence`
columns for download.

## Files

| File | Description |
|------|-------------|
| `Twitter sentiment analysis/sentiment_analysis.ipynb` | Full notebook: EDA, cleaning, training, evaluation |
| `Twitter sentiment analysis/app.py` | Streamlit web app |
| `Twitter sentiment analysis/models/lr_model.pkl` | Saved Logistic Regression model |
| `Twitter sentiment analysis/models/tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `tweets.csv` | Sentiment140 dataset |

## Key Takeaway

Tweets with personal anecdotes, exclamations, and positive words predict positive sentiment reliably.
Short angry tweets with negations are strong signals of negativity. TF-IDF bigrams capture phrases like
"not good" or "love this" that single words miss.
