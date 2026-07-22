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


def test_non_english_input_gets_translated_then_classified():
    """
    New behavior: non-English input is translated to English, then
    actually run through classify/sentiment/themes -- not just flagged
    and skipped. Still flagged afterward, but for a different, honest
    reason (translation happened, worth a spot-check), and with a real
    classification result attached instead of a placeholder.
    """
    class TranslatingClient:
        def complete(self, system, user, temperature=0.0):
            if "translator" in system.lower():
                return "The app crashes every time I upload a profile photo."
            if "category" in system.lower() and "sentiment" not in system.lower():
                return '{"category": "Bug / Technical Issue", "confidence": 0.9}'
            if "urgency" in system.lower():
                return '{"sentiment": "Negative", "sentiment_score": -0.8, "urgency": "High"}'
            return '["photo upload crash"]'

    item = FeedbackItem(id="fb2", text="La aplicacion se bloquea cada vez que subo una foto")
    result = process_one(item, TranslatingClient(), logger=lambda msg: None)

    assert result.classification.category == Category.BUG
    assert len(result.themes) > 0
    assert result.feedback.text == item.text  # original text preserved, not the translation
    assert result.flagged_for_review is True
    assert "Auto-translated" in result.review_reason


def test_non_english_translation_failure_falls_back_to_skip():
    """If translation itself fails (API outage, etc.), fall back to the
    original honest skip -- never classify an untranslated or garbled text."""

    class ExplodingClient:
        def complete(self, *a, **k):
            raise ConnectionError("simulated outage")

    item = FeedbackItem(id="fb2b", text="La aplicacion se bloquea cada vez que subo una foto")
    result = process_one(item, ExplodingClient(), logger=lambda msg: None)
    assert result.flagged_for_review is True
    assert "translation failed" in result.review_reason
    assert result.classification.category == Category.OTHER


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


def test_processing_time_reflects_real_elapsed_time(monkeypatch):
    """
    processing_time_ms must be REAL measured wall-clock time, not a
    fabricated placeholder -- this is what the dashboard's business-impact
    metric relies on being honest. Proven here by injecting a known,
    deliberate delay into the fake client and confirming the measured
    time is at least that long.
    """
    import time as time_module

    class SlowScriptedAIClient(ScriptedAIClient):
        def complete(self, system, user, temperature=0.0):
            time_module.sleep(0.05)
            return super().complete(system, user, temperature)

    fake = SlowScriptedAIClient(
        classification_response='{"category": "Bug / Technical Issue", "confidence": 0.9}',
        sentiment_response='{"sentiment": "Negative", "sentiment_score": -0.7, "urgency": "High"}',
    )
    item = FeedbackItem(id="fb5", text="The app crashes every time I try to upload a photo here")
    result = process_one(item, fake, logger=lambda msg: None)

    # 3 calls (classify, sentiment, themes) x ~50ms sleep each = ~150ms minimum
    assert result.processing_time_ms >= 140


def test_processing_time_is_none_when_brain_layer_is_skipped():
    """Non-English input skips classify/sentiment/themes entirely -- the
    timing field should honestly reflect that no AI processing happened
    (None), not a misleading 0 that implies instant processing."""

    class ExplodingClient:
        def complete(self, *a, **k):
            raise AssertionError("brain layer should not be called")

    item = FeedbackItem(id="fb6", text="La aplicacion se bloquea cada vez que subo una foto")
    result = process_one(item, ExplodingClient(), logger=lambda msg: None)
    assert result.processing_time_ms is None