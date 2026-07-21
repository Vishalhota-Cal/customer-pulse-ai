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
    feedback: FeedbackItem, client: AIClient, logger=print
) -> ProcessedFeedback:
    """Run the full classify -> sentiment -> themes sequence for ONE item."""

    # Edge case: non-English input. The few-shot examples in every brain/
    # module are English-only, so running non-English text through them
    # wouldn't error -- it would just quietly produce an unreliable guess.
    # Detecting and flagging is safer than pretending confidence in a
    # result the classifier was never taught to give.
    if not feedback.is_blank:
        detected_lang = _detect_non_english(feedback.text)
        if detected_lang:
            logger(
                f"[pipeline] feedback_id={feedback.id} flagged: "
                f"detected non-English content (lang={detected_lang})"
            )
            return _placeholder_result(
                feedback,
                reason=f"Detected non-English content (lang={detected_lang}); "
                f"not run through the English-only classifier.",
            )

    step_times = {}

    t0 = time.perf_counter()
    classification = classify_feedback(feedback, client)
    step_times["classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sentiment = score_sentiment(feedback, client)
    step_times["sentiment"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    themes = extract_themes(feedback, client)
    step_times["themes"] = time.perf_counter() - t0

    rounded_timings = {k: round(v, 3) for k, v in step_times.items()}
    logger(f"[pipeline] feedback_id={feedback.id} timings={rounded_timings}")

    flagged = classification.confidence < LOW_CONFIDENCE_THRESHOLD
    reason = (
        f"Low classifier confidence ({classification.confidence}); needs human review."
        if flagged
        else None
    )

    return ProcessedFeedback(
        feedback=feedback,
        classification=classification,
        sentiment=sentiment,
        themes=themes,
        flagged_for_review=flagged,
        review_reason=reason,
    )


def run_pipeline(
    feedback_items: list[FeedbackItem],
    client: AIClient,
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
            processed = process_one(item, client, logger=logger)
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