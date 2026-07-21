"""Tests for brain/classifier.py -- uses ScriptedAIClient, never a real API call."""

from domain.feedback import Category, FeedbackItem
from brain.classifier import classify_feedback
from tests.fakes import ExplodingAIClient, ScriptedAIClient


def test_blank_input_never_calls_the_api():
    fake = ScriptedAIClient()
    result = classify_feedback(FeedbackItem(id="fb0", text="   "), fake)
    assert result.category == Category.OTHER
    assert result.confidence == 1.0
    assert fake.call_count == 0


def test_valid_response_parsed_correctly():
    fake = ScriptedAIClient(
        classification_response='{"category": "Bug / Technical Issue", "confidence": 0.93}'
    )
    result = classify_feedback(FeedbackItem(id="fb1", text="it crashes"), fake)
    assert result.category == Category.BUG
    assert result.confidence == 0.93


def test_markdown_fenced_response_still_parses():
    fake = ScriptedAIClient(
        classification_response='```json\n{"category": "Positive Feedback / Praise", "confidence": 0.88}\n```'
    )
    result = classify_feedback(FeedbackItem(id="fb2", text="I love it"), fake)
    assert result.category == Category.PRAISE


def test_invalid_category_falls_back_safely():
    fake = ScriptedAIClient(classification_response='{"category": "Not A Real Category", "confidence": 0.9}')
    result = classify_feedback(FeedbackItem(id="fb3", text="weird"), fake)
    assert result.category == Category.OTHER
    assert result.confidence == 0.0


def test_malformed_json_falls_back_safely():
    fake = ScriptedAIClient(classification_response="not json")
    result = classify_feedback(FeedbackItem(id="fb4", text="whatever"), fake)
    assert result.category == Category.OTHER
    assert result.confidence == 0.0


def test_api_outage_does_not_crash():
    result = classify_feedback(FeedbackItem(id="fb5", text="does this crash?"), ExplodingAIClient())
    assert result.category == Category.OTHER
    assert result.confidence == 0.0


def test_repeated_classification_is_consistent():
    fake = ScriptedAIClient(classification_response='{"category": "Performance / Speed", "confidence": 0.85}')
    item = FeedbackItem(id="fb6", text="slow to load")
    assert classify_feedback(item, fake) == classify_feedback(item, fake)