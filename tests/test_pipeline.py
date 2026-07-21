"""Tests for orchestration/pipeline.py."""

from pathlib import Path

import orchestration.pipeline as pipeline_module
from domain.feedback import Category, FeedbackItem
from orchestration.pipeline import process_one, run_pipeline
from persistence import store
from tests.fakes import ScriptedAIClient


def test_normal_english_item_runs_full_pipeline():
    fake = ScriptedAIClient(
        classification_response='{"category": "Bug / Technical Issue", "confidence": 0.9}',
        sentiment_response='{"sentiment": "Negative", "sentiment_score": -0.7, "urgency": "High"}',
        theme_response='["upload crash"]',
    )
    item = FeedbackItem(id="fb1", text="The app crashes every time I try to upload a photo here")
    result = process_one(item, fake, logger=lambda msg: None)
    assert result.classification.category == Category.BUG
    assert result.flagged_for_review is False


def test_non_english_input_is_flagged_without_calling_brain_layer():
    class ExplodingClient:
        def complete(self, *a, **k):
            raise AssertionError("brain layer should NOT be called for non-English input")

    item = FeedbackItem(id="fb2", text="La aplicacion se bloquea cada vez que subo una foto")
    result = process_one(item, ExplodingClient(), logger=lambda msg: None)
    assert result.flagged_for_review is True
    assert "non-English" in result.review_reason


def test_low_confidence_classification_is_flagged():
    fake = ScriptedAIClient(
        classification_response='{"category": "Other / Uncategorised", "confidence": 0.2}'
    )
    item = FeedbackItem(id="fb3", text="a somewhat ambiguous piece of feedback about something")
    result = process_one(item, fake, logger=lambda msg: None)
    assert result.flagged_for_review is True


def test_batch_run_persists_to_the_given_store_path(tmp_path):
    test_store = tmp_path / "pipeline_test_store.jsonl"
    fake = ScriptedAIClient(
        classification_response='{"category": "Bug / Technical Issue", "confidence": 0.9}',
        sentiment_response='{"sentiment": "Negative", "sentiment_score": -0.7, "urgency": "High"}',
    )
    items = [
        FeedbackItem(id="fb1", text="The app crashes every time I try to upload a photo here"),
        FeedbackItem(id="fb2", text="a somewhat ambiguous piece of feedback about something"),
    ]
    run_pipeline(items, fake, save=True, store_path=test_store, logger=lambda msg: None)
    loaded = store.load_all(test_store)
    assert len(loaded) == 2


def test_orchestration_safety_net_catches_unexpected_exceptions(monkeypatch):
    def exploding_classifier(feedback, client):
        raise RuntimeError("simulated unexpected bug")

    monkeypatch.setattr(pipeline_module, "classify_feedback", exploding_classifier)

    item = FeedbackItem(id="fb4", text="some totally normal english feedback text")
    results = run_pipeline([item], ScriptedAIClient(), save=False, logger=lambda msg: None)

    assert len(results) == 1
    assert results[0].flagged_for_review is True
    assert "Unexpected pipeline error" in results[0].review_reason