import streamlit as st
import pandas as pd
from textblob import TextBlob
import seaborn as sns
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
    if polarity >= 0.03:
        return "Positive"
    elif polarity <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["translated_text"].apply(classify_sentiment)
df["subjectivity"] = df["translated_text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
df["flagged"] = df["subjectivity"] > 0.85

# --- Sidebar filters ---
# Sidebar Filters
st.sidebar.title("🔎 Filter Comments")

# Date range filter
date_range = st.sidebar.date_input("Select date range", [df["date"].min(), df["date"].max()])
filtered_df = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# Sentiment filter
sentiment_options = ["All", "Positive", "Negative", "Neutral"]
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

# Keyword search filter
keyword = st.sidebar.text_input("Search keyword in comments")
if keyword:
    filtered_df = filtered_df[filtered_df["translated_text"].str.contains(keyword, case=False, na=False)]

# Flagged-only checkbox
show_flagged = st.sidebar.checkbox("Show only highly subjective comments")
if show_flagged:
    filtered_df = filtered_df[filtered_df["flagged"]]

# Drop exact duplicates based on comment (ignoring date)
# filtered_df = filtered_df.drop_duplicates(subset="translated_text")


# --- Sentiment Distribution (Bar & Pie) ---
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Sentiment Distribution (Pie Chart)")
fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['lightgreen', 'lightcoral', 'lightgrey'][:len(sentiment_counts)]
)
ax.axis('equal')
st.pyplot(fig)

# --- Sentiment Trend Over Time ---
st.subheader("Sentiment Trend Over Time")
sentiment_over_time = (
    filtered_df.groupby([filtered_df["date"].dt.to_period("W"), "sentiment"])
    .size().unstack(fill_value=0)
)
sentiment_over_time.index = sentiment_over_time.index.to_timestamp()
st.line_chart(sentiment_over_time)

# --- Sentiment Heatmap ---
st.subheader("Sentiment Heatmap (Weekly %)")
sentiment_counts_week = (
    filtered_df.groupby([filtered_df["date"].dt.to_period("W"), "sentiment"])
    .size().unstack(fill_value=0)
)
# Convert to percentages per week
sentiment_percentages = sentiment_counts_week.div(sentiment_counts_week.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(
    sentiment_percentages.T,
    cmap="coolwarm",
    cbar_kws={'label': 'Percentage (%)'}
)
ax.set_xlabel("Week")
ax.set_ylabel("Sentiment")
st.pyplot(fig)

# --- Subjectivity Distribution ---
st.subheader("Subjectivity Score Distribution")
fig, ax = plt.subplots(figsize=(6,4))
filtered_df["subjectivity"].hist(bins=20, ax=ax)
ax.set_title("Subjectivity Score Distribution")
ax.set_xlabel("Subjectivity")
ax.set_ylabel("Number of Comments")
st.pyplot(fig)

# --- Word Cloud (All Comments) ---
st.subheader("Top Keywords (Word Cloud)")
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "rt", "https", "t", "co", "rwanda", "public", "fare", "city",
    "kigali", "kololo", "whole", "potholes", "consider", "new",
    "half", "preeminent", "diplomatic", "fair"
])
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

text_corpus = ' '.join(filtered_df["translated_text"].dropna().apply(clean_text).tolist())
wc = WordCloud(
    width=1200, height=600,
    background_color='white',
    stopwords=custom_stopwords,
    max_words=150,
    collocations=False,
    colormap='viridis',
    contour_width=2,
    contour_color='steelblue'
).generate(text_corpus)

fig = plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Top Keywords in Public Transport Comments", fontsize=16, pad=20)
st.pyplot(fig)

# --- Word Clouds by Sentiment ---
st.subheader("Word Clouds by Sentiment")
cols = st.columns(3)
for idx, sentiment in enumerate(["Positive", "Negative", "Neutral"]):
    subset = filtered_df[filtered_df["sentiment"] == sentiment]
    text = ' '.join(subset["translated_text"].dropna().apply(clean_text))
    wc_sentiment = WordCloud(width=400, height=400, stopwords=custom_stopwords).generate(text)
    fig_sent, ax_sent = plt.subplots(figsize=(4, 4))
    ax_sent.imshow(wc_sentiment, interpolation='bilinear')
    ax_sent.axis('off')
    ax_sent.set_title(f"{sentiment} Comments")
    cols[idx].pyplot(fig_sent)

# --- Sample Comments ---

filtered_df = filtered_df.drop_duplicates(subset="translated_text")

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
- **{flagged}** comments were flagged as **highly subjective** (subjectivity > 0.85), which may indicate confusion or misinformation.
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
- ✅ **Improve communication** around fare structure and calculations.git status

- 📢 **Use SMS, community radio, and bus posters** to educate the public.
- 💳 **Standardize smart card fare updates** to avoid confusion.
- 🧾 **Review flagged comments weekly** for emerging concerns.
""")


