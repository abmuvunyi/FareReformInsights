# app.py — FareReformInsights Streamlit Dashboard

import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(page_title="FareReformInsights", layout="wide")

# --- Load data ---
st.title("📊 FareReformInsights – Public Sentiment Dashboard")
df = pd.read_csv("data/fare_sentiment_sample.csv")
df["date"] = pd.to_datetime(df["date"])

# --- Analyze Sentiment ---
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        return "Positive"
    elif polarity <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["text"].apply(classify_sentiment)
df["subjectivity"] = df["text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["flagged"] = df["subjectivity"] > 0.85

# --- Sidebar filters ---
st.sidebar.title("🔍 Filters")
date_range = st.sidebar.date_input("Select date range", [df["date"].min(), df["date"].max()])
filtered_df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

# --- Sentiment Over Time ---
st.subheader("Sentiment Trend Over Time")
sentiment_over_time = (
    filtered_df.groupby([filtered_df["date"].dt.to_period("W"), "sentiment"])
    .size()
    .unstack(fill_value=0)
)
sentiment_over_time.index = sentiment_over_time.index.to_timestamp()
st.line_chart(sentiment_over_time)

# --- Word Cloud ---
st.subheader("Common Words in Comments")
word_text = ' '.join(filtered_df["text"].dropna())
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(word_text)
st.image(wordcloud.to_array(), use_column_width=True)

# --- Sample Comments ---
st.subheader("Sample Comments")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Positive Comments**")
    for comment in filtered_df[filtered_df["sentiment"] == "Positive"]["text"].head(3):
        st.success(comment)

with col2:
    st.markdown("**Top Negative Comments**")
    for comment in filtered_df[filtered_df["sentiment"] == "Negative"]["text"].head(3):
        st.error(comment)

# --- Flagged Comments ---
st.subheader("⚠️ Flagged Comments (High Subjectivity)")
for _, row in filtered_df[filtered_df["flagged"]].iterrows():
    st.warning(f"{row['date'].date()}: {row['text']}")
