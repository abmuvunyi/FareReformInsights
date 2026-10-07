import pytest
from utils import analyze_sentiment, clean_text

def test_analyze_sentiment():
    # Positive sentiment
    sentiment, subj, flagged = analyze_sentiment("I love this new public transport system!")
    assert sentiment == "Positive"
    assert not flagged

    # Negative sentiment
    sentiment, subj, flagged = analyze_sentiment("The prices are terribly expensive and service is bad.")
    assert sentiment == "Negative"

    # Neutral sentiment
    sentiment, subj, flagged = analyze_sentiment("The bus arrives at 8 AM.")
    assert sentiment == "Neutral"

    # Highly subjective
    sentiment, subj, flagged = analyze_sentiment("I am extremely furious and completely outraged by this incredibly awful service!!!")
    assert flagged is True

def test_clean_text():
    # URL removal
    assert clean_text("Check this link https://rwanda.gov/transport") == "check this link"

    # Mention removal
    assert clean_text("Hello @abmuvunyi what do you think?") == "hello  what do you think"

    # Special characters removal
    assert clean_text("Bus fare is $10.00!") == "bus fare is"

    # Non-string input
    assert clean_text(None) == ""
    assert clean_text(123) == ""
