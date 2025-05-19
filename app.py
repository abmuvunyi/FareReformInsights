import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter

st.set_page_config(page_title="FareReformInsights", layout="wide")

# --- Load data ---
st.title("FareReformInsights – Public Sentiment Dashboard")
df = pd.read_csv("data/combined_sentiment_dataset.csv")
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# --- Analyze Sentiment & Subjectivity ---
def classify_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity >= 0.05:
        return "Positive"
    elif polarity <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["translated_text"].apply(classify_sentiment)
df["subjectivity"] = df["translated_text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
df["flagged"] = df["subjectivity"] > 0.85

# --- Sidebar filters ---
st.sidebar.title("Filters")
date_range = st.sidebar.date_input("Select date range", [df["date"].min(), df["date"].max()])
filtered_df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Sentiment Distribution (Pie Chart)")
fig, ax = plt.subplots()
ax.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightgreen', 'lightcoral', 'lightgrey'][:len(sentiment_counts)]
)
ax.axis('equal')
st.pyplot(fig)

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
st.subheader("Most Common Meaningful Words in Comments")

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "rt", "https", "t", "co", "rwanda", "transport", "public", "fare",
    "city", "kigali", "kololo", "whole", "potholes", "consider",
    "preeminent", "diplomatic", "eminent", "combined"
])

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

word_text = ' '.join(filtered_df["translated_text"].dropna().apply(clean_text).tolist())
wordcloud = WordCloud(
    stopwords=custom_stopwords,
    background_color='white',
    max_words=100
).generate(word_text)

st.image(wordcloud.to_array(), use_column_width=True)

# --- Sample Comments ---
st.subheader("Sample Comments")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top Positive Comments**")
    for comment in filtered_df[filtered_df["sentiment"] == "Positive"]["translated_text"].head(3):
        st.success(comment)

with col2:
    st.markdown("**Top Negative Comments**")
    for comment in filtered_df[filtered_df["sentiment"] == "Negative"]["translated_text"].head(3):
        st.error(comment)

# --- Flagged Comments ---
st.subheader("⚠️ Flagged Comments (Highly Subjective)")
for _, row in filtered_df[filtered_df["flagged"]].iterrows():
    st.warning(f"{row['date'].date()}: {row['translated_text']}")

# --- Insights Summary ---
st.subheader("📌 Key Insights")

total = len(filtered_df)
pos = (filtered_df["sentiment"] == "Positive").sum()
neg = (filtered_df["sentiment"] == "Negative").sum()
neu = (filtered_df["sentiment"] == "Neutral").sum()
flagged = filtered_df["flagged"].sum()

st.markdown(f"""
- Out of **{total}** comments, **{pos}** were positive (**{(pos/total)*100:.1f}%**), **{neg}** negative (**{(neg/total)*100:.1f}%**), and **{neu}** neutral.
- **{flagged}** comments were flagged as **highly subjective**, which may indicate confusion, frustration, or misinformation.
""")

# --- Common Topic Highlights ---
st.subheader("🔎 Common Keywords")
keywords = ["fare", "increase", "card", "price", "cheap", "expensive", "distance", "smart", "explain", "confusing"]
tokens = ' '.join(filtered_df["translated_text"].dropna()).lower().split()
found = [w for w in tokens if w in keywords]
top_keywords = Counter(found).most_common(5)

if top_keywords:
    st.markdown("**Frequent keywords in concerns:** " + ", ".join([k[0] for k in top_keywords]))
else:
    st.markdown("_No dominant keywords found._")

# --- Recommendations ---
st.subheader("🧠 Recommendations")
st.markdown("""
- ✅ **Improve communication** around fare structure and calculations.
- 📢 **Use SMS, community radio, and bus posters** to educate the public.
- 💳 **Standardize smart card fare updates** to avoid confusion and price mismatch.
- 🧾 **Review flagged comments weekly** to spot emotional concerns or misinformation early.
""")

# --- Export Flagged Comments ---
st.download_button(
    "📥 Download Flagged Comments as CSV",
    data=filtered_df[filtered_df["flagged"]][["date", "translated_text"]].to_csv(index=False),
    file_name="flagged_comments.csv",
    mime="text/csv"
)
