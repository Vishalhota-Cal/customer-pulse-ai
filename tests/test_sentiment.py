"""Tests for brain/sentiment.py -- uses ScriptedAIClient, never a real API call."""

from domain.feedback import FeedbackItem, Sentiment, Urgency
from brain.sentiment import score_sentiment
from tests.fakes import ExplodingAIClient, ScriptedAIClient


def test_blank_input_returns_neutral_without_calling_api():
    fake = ScriptedAIClient()
    result = score_sentiment(FeedbackItem(id="fb0", text=""), fake)
    assert result.sentiment == Sentiment.NEUTRAL
    assert result.sentiment_score == 0.0
    assert fake.call_count == 0


def test_valid_response_parsed_correctly():
    fake = ScriptedAIClient(
        sentiment_response='{"sentiment": "Negative", "sentiment_score": -0.8, "urgency": "High"}'
    )
    result = score_sentiment(FeedbackItem(id="fb1", text="charged twice, fix now"), fake)
    assert result.sentiment == Sentiment.NEGATIVE
    assert result.urgency == Urgency.HIGH


def test_malformed_json_falls_back_to_neutral():
    fake = ScriptedAIClient(sentiment_response="not json")
    result = score_sentiment(FeedbackItem(id="fb2", text="whatever"), fake)
    assert result.sentiment == Sentiment.NEUTRAL
    assert result.sentiment_score == 0.0


def test_api_outage_does_not_crash():
    result = score_sentiment(FeedbackItem(id="fb3", text="does this crash?"), ExplodingAIClient())
    assert result.sentiment == Sentiment.NEUTRAL


def test_out_of_range_score_from_model_is_caught_and_falls_back():
    # Domain-layer validation should reject this at construction, and
    # score_sentiment's broad except should catch that and fall back
    # safely rather than propagating the error.
    fake = ScriptedAIClient(sentiment_response='{"sentiment": "Negative", "sentiment_score": -5.0, "urgency": "High"}')
    result = score_sentiment(FeedbackItem(id="fb4", text="edge case"), fake)
    assert result.sentiment == Sentiment.NEUTRAL
    assert result.sentiment_score == 0.0


def test_repeated_scoring_is_consistent():
    fake = ScriptedAIClient(sentiment_response='{"sentiment": "Negative", "sentiment_score": -0.3, "urgency": "Medium"}')
    item = FeedbackItem(id="fb5", text="minor layout issue")
    assert score_sentiment(item, fake) == score_sentiment(item, fake)