"""Tests for brain/summary_generator.py."""

from domain.feedback import (
    Category,
    ClassificationResult,
    FeedbackItem,
    ProcessedFeedback,
    Sentiment,
    SentimentResult,
    ThemeTag,
    Urgency,
)
from brain.summary_generator import _compute_stats, generate_weekly_summary
from tests.fakes import ExplodingAIClient, ScriptedAIClient


def _make_item(id_, text, category, sentiment, score, urgency, themes=None):
    f = FeedbackItem(id=id_, text=text)
    c = ClassificationResult(feedback_id=id_, category=category, confidence=0.9)
    s = SentimentResult(feedback_id=id_, sentiment=sentiment, sentiment_score=score, urgency=urgency)
    t = [ThemeTag(feedback_id=id_, theme=th) for th in (themes or [])]
    return ProcessedFeedback(feedback=f, classification=c, sentiment=s, themes=t)


def _sample_batch():
    return [
        _make_item("1", "App crashes on upload", Category.BUG, Sentiment.NEGATIVE, -0.8, Urgency.HIGH, ["photo upload crash"]),
        _make_item("2", "Charged twice this month", Category.BILLING, Sentiment.NEGATIVE, -0.7, Urgency.HIGH, ["duplicate charge"]),
        _make_item("3", "Love the new design", Category.PRAISE, Sentiment.POSITIVE, 0.9, Urgency.LOW, ["positive redesign feedback"]),
        _make_item("4", "App crashes when uploading too", Category.BUG, Sentiment.NEGATIVE, -0.75, Urgency.HIGH, ["photo upload crash"]),
        _make_item("5", "Wish there was dark mode", Category.FEATURE_REQUEST, Sentiment.NEUTRAL, 0.1, Urgency.LOW, ["dark mode request"]),
    ]


def test_empty_batch_returns_honest_message_without_api_call():
    fake = ScriptedAIClient()
    result = generate_weekly_summary([], fake)
    assert "No feedback" in result
    assert fake.call_count == 0


def test_stats_computation_is_accurate():
    stats = _compute_stats(_sample_batch())
    assert stats["total_items"] == 5
    assert stats["urgent_count"] == 3  # items 1, 2, 4
    assert stats["category_counts"]["Bug / Technical Issue"] == 2


def test_ai_generated_summary_uses_real_computed_stats():
    fake = ScriptedAIClient(summary_response="Sentiment was negative, driven by crash reports.")
    result = generate_weekly_summary(_sample_batch(), fake)
    assert "crash" in result.lower()
    assert fake.call_count == 1
    # Confirm the real stats actually reached the prompt, not just that
    # SOME response came back.
    assert "photo upload crash" in fake.last_user
    assert "HIGH urgency items: 3" in fake.last_user


def test_api_failure_returns_honest_templated_fallback():
    result = generate_weekly_summary(_sample_batch(), ExplodingAIClient())
    assert "fallback" in result.lower()
    assert "photo upload crash" in result