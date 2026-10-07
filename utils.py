from textblob import TextBlob
import re

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text.
    Returns a tuple of (sentiment_label, subjectivity_score, flagged).
    """
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity >= 0.03:
        sentiment = "Positive"
    elif polarity <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    flagged = subjectivity > 0.85

    return sentiment, subjectivity, flagged

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()
