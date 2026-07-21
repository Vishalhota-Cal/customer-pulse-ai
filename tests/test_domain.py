"""Tests for domain/feedback.py -- validation rules and data shapes."""

import pytest

from domain.feedback import (
    AggregatedTheme,
    Category,
    ClassificationResult,
    FeedbackItem,
    ProcessedFeedback,
    Sentiment,
    SentimentResult,
    ThemeTag,
    Urgency,
)


def test_normal_feedback_item_construction():
    f = FeedbackItem(id="fb1", text="Hello world", source="review")
    assert f.id == "fb1"
    assert f.is_blank is False


def test_whitespace_only_text_is_blank():
    f = FeedbackItem(id="fb2", text="   \n  ")
    assert f.is_blank is True


def test_oversized_text_rejected():
    with pytest.raises(ValueError):
        FeedbackItem(id="fb3", text="x" * 6000)


def test_invalid_category_string_rejected():
    with pytest.raises(ValueError):
        Category("Buggg")


def test_classification_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        ClassificationResult(feedback_id="fb1", category=Category.BUG, confidence=1.5)


def test_sentiment_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        SentimentResult(
            feedback_id="fb1",
            sentiment=Sentiment.NEGATIVE,
            sentiment_score=-5.0,
            urgency=Urgency.HIGH,
        )


def test_blank_theme_tag_rejected():
    with pytest.raises(ValueError):
        ThemeTag(feedback_id="fb1", theme="   ")


def test_aggregated_theme_requires_positive_count():
    with pytest.raises(ValueError):
        AggregatedTheme(label="something", count=0)


def test_processed_feedback_full_object_graph():
    f = FeedbackItem(id="fb1", text="crashes on upload")
    c = ClassificationResult(feedback_id="fb1", category=Category.BUG, confidence=0.9)
    s = SentimentResult(
        feedback_id="fb1", sentiment=Sentiment.NEGATIVE, sentiment_score=-0.7, urgency=Urgency.HIGH
    )
    t = ThemeTag(feedback_id="fb1", theme="upload crash")
    p = ProcessedFeedback(feedback=f, classification=c, sentiment=s, themes=[t])
    assert p.classification.category == Category.BUG
    assert p.flagged_for_review is False  # default