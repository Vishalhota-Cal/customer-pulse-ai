"""
Orchestration layer -- the ONLY place that calls the brain-layer pieces
(classifier, sentiment, theme extractor) in order, times each step, logs
what happened, and decides whether an item needs to be flagged for human
review. No classification/sentiment/theme LOGIC lives here -- this file
just sequences and coordinates calls to brain/*.py, and hands the result
to persistence/store.py.
"""

from __future__ import annotations

import time

from brain.classifier import AIClient, classify_feedback
from brain.sentiment import score_sentiment
from brain.theme_aggregator import extract_themes
from brain.translator import translate_to_english
from orchestration.alerting import send_urgent_alert
from domain.feedback import (
    Category,
    ClassificationResult,
    FeedbackItem,
    ProcessedFeedback,
    Sentiment,
    SentimentResult,
    Urgency,
)
from persistence import store as persistence_store

# Below this classifier confidence, treat the result as unreliable enough
# to need a human look -- this threshold is what turns "the AI answered
# something" into "the AI answered something we should actually trust."
LOW_CONFIDENCE_THRESHOLD = 0.4

# langdetect is an optional dependency: if it's missing for any reason,
# the pipeline degrades gracefully (skips language detection) rather than
# crashing on import. This matters because language detection is a nice-
# to-have edge case handler, not core pipeline functionality -- losing it
# should never take down classification/sentiment/themes.
try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic detection results across runs
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def _detect_non_english(text: str) -> str | None:
    """
    Return a language code if text is confidently non-English, else None.

    Short strings ("ok", "meh", "5 stars") are skipped deliberately --
    language detection on very short text is unreliable and would flag
    plenty of perfectly fine English feedback as a false positive.
    """
    if not _LANGDETECT_AVAILABLE or len(text.strip()) < 10:
        return None
    try:
        lang = detect(text)
    except LangDetectException:
        return None
    return lang if lang != "en" else None


def _placeholder_result(
    feedback: FeedbackItem, reason: str
) -> ProcessedFeedback:
    """
    Shared helper for the two places this file needs to produce a safe,
    flagged, do-nothing-clever result: non-English input, and an
    unexpected error the brain layer somehow didn't already catch.
    """
    return ProcessedFeedback(
        feedback=feedback,
        classification=ClassificationResult(
            feedback_id=feedback.id, category=Category.OTHER, confidence=0.0
        ),
        sentiment=SentimentResult(
            feedback_id=feedback.id,
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.0,
            urgency=Urgency.LOW,
        ),
        themes=[],
        flagged_for_review=True,
        review_reason=reason,
    )


def process_one(
    feedback: FeedbackItem, client: AIClient, escalation_client: AIClient | None = None, logger=print
) -> ProcessedFeedback:
    """Run the full classify -> sentiment -> themes sequence for ONE item."""

    # Edge case: non-English input. The few-shot examples in every brain/
    # module are English-only, so classifying the ORIGINAL non-English
    # text would just quietly produce an unreliable guess. Instead of
    # skipping it entirely, translate to English first, then run the
    # normal pipeline on the translation. Still flagged for review
    # afterward -- not because it was skipped, but because translation
    # quality is a real, honest extra source of uncertainty worth a
    # human spot-check.
    working_feedback = feedback
    translation_note = None
    if not feedback.is_blank:
        detected_lang = _detect_non_english(feedback.text)
        if detected_lang:
            translated_text = translate_to_english(feedback.text, client)
            if translated_text is None:
                # Translation itself failed -- fall back to the old,
                # honest skip behavior rather than classifying gibberish
                # or the untranslated original.
                logger(
                    f"[pipeline] feedback_id={feedback.id} flagged: "
                    f"detected non-English (lang={detected_lang}), translation failed"
                )
                return _placeholder_result(
                    feedback,
                    reason=f"Detected non-English content (lang={detected_lang}); "
                    f"translation failed, not run through the classifier.",
                )
            logger(
                f"[pipeline] feedback_id={feedback.id} translated from "
                f"lang={detected_lang} before classification"
            )
            # Build a new FeedbackItem carrying the TRANSLATED text through
            # the rest of the pipeline, while preserving the original id/
            # source/timestamp -- domain validation (length cap, etc.)
            # still applies to the translated text like any other input.
            working_feedback = FeedbackItem(
                id=feedback.id,
                text=translated_text,
                source=feedback.source,
                submitted_at=feedback.submitted_at,
            )
            translation_note = (
                f"Auto-translated from lang={detected_lang} before classification; "
                f"verify translation accuracy."
            )

    step_times = {}

    t0 = time.perf_counter()
    classification = classify_feedback(working_feedback, client, escalation_client=escalation_client)
    step_times["classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sentiment = score_sentiment(working_feedback, client)
    step_times["sentiment"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    themes = extract_themes(working_feedback, client)
    step_times["themes"] = time.perf_counter() - t0

    rounded_timings = {k: round(v, 3) for k, v in step_times.items()}
    logger(f"[pipeline] feedback_id={feedback.id} timings={rounded_timings}")

    total_ms = sum(step_times.values()) * 1000  # perf_counter deltas are in seconds

    low_confidence = classification.confidence < LOW_CONFIDENCE_THRESHOLD
    flagged = low_confidence or translation_note is not None
    reason_parts = []
    if translation_note:
        reason_parts.append(translation_note)
    if low_confidence:
        reason_parts.append(f"Low classifier confidence ({classification.confidence}); needs human review.")
    reason = " ".join(reason_parts) if reason_parts else None

    if sentiment.urgency == Urgency.HIGH:
        # Best-effort: send_urgent_alert() itself never raises, and is a
        # silent no-op if no webhook URL is configured. This never blocks
        # or delays returning the result -- a failed notification should
        # never be the reason feedback processing appears to hang.
        send_urgent_alert(
            feedback_id=feedback.id,
            category=classification.category.value,
            text=feedback.text,
        )

    return ProcessedFeedback(
        feedback=feedback,  # ORIGINAL feedback (untranslated text) -- this is what
                            # the customer actually wrote; the translation was an
                            # internal step to make classification possible, not a
                            # replacement for the real record.
        classification=classification,
        sentiment=sentiment,
        themes=themes,
        flagged_for_review=flagged,
        review_reason=reason,
        processing_time_ms=total_ms,
    )


def run_pipeline(
    feedback_items: list[FeedbackItem],
    client: AIClient,
    escalation_client: AIClient | None = None,
    save: bool = True,
    store_path=None,
    logger=print,
) -> list[ProcessedFeedback]:
    """
    Run the full pipeline over a batch of feedback items, in order,
    logging timing per item and an overall summary at the end.

    store_path defaults to the persistence layer's own default location,
    resolved at CALL time (not import time) so overriding
    persistence_store.DEFAULT_STORE_PATH actually takes effect -- required
    for tests to write to an isolated test file instead of silently
    touching production data.
    """
    results: list[ProcessedFeedback] = []
    start = time.perf_counter()
    resolved_path = store_path if store_path is not None else persistence_store.DEFAULT_STORE_PATH

    for item in feedback_items:
        try:
            processed = process_one(item, client, escalation_client=escalation_client, logger=logger)
        except Exception as e:
            # This should be unreachable -- every brain/*.py function is
            # written to never raise. It exists anyway as an orchestration-
            # level safety net: if a future change to any brain module ever
            # starts raising unexpectedly, ONE bad item still shouldn't
            # take down the entire batch run.
            logger(
                f"[pipeline] UNEXPECTED error processing feedback_id={item.id}: {e}"
            )
            processed = _placeholder_result(
                item, reason=f"Unexpected pipeline error: {e}"
            )
        results.append(processed)
        if save:
            persistence_store.save(processed, path=resolved_path)

    elapsed = time.perf_counter() - start
    logger(f"[pipeline] Processed {len(results)} item(s) in {elapsed:.2f}s")
    return results