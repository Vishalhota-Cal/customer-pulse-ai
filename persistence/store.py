"""
Persistence layer — the ONLY file in this project allowed to touch storage.

Nothing else imports json-file-reading/writing logic or a database driver
directly. If another layer needs data saved or loaded, it calls a function
in here — it never opens a file itself.

Storage format: JSON Lines (.jsonl), one ProcessedFeedback record per line.
Why JSONL instead of one big JSON array:
  - Appending a new record is just "write one more line" — no read-modify-
    rewrite-the-whole-file dance, which matters once you're processing
    hundreds of feedback items and don't want O(n) disk I/O per item.
  - A crash mid-write corrupts at most the last line, not the entire file.
  - Proportionate for a 2-week demo project — a real database would be
    overkill here, but a single unstructured file with no append safety
    would be worse.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

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

DEFAULT_STORE_PATH = Path("data/processed_feedback.jsonl")


# ---------------------------------------------------------------------------
# Serialization: domain object -> plain dict (JSON-safe)
# ---------------------------------------------------------------------------
# These conversions live here, not in the domain layer, on purpose — the
# domain layer shouldn't know or care that JSON is the storage format.
# If we ever swap to a real database, only this file changes.

def _feedback_item_to_dict(item: FeedbackItem) -> dict:
    return {
        "id": item.id,
        "text": item.text,
        "submitted_at": item.submitted_at.isoformat(),
        "source": item.source,
    }


def _feedback_item_from_dict(d: dict) -> FeedbackItem:
    return FeedbackItem(
        id=d["id"],
        text=d["text"],
        submitted_at=datetime.fromisoformat(d["submitted_at"]),
        source=d["source"],
    )


def _classification_to_dict(c: ClassificationResult) -> dict:
    return {
        "feedback_id": c.feedback_id,
        "category": c.category.value,
        "confidence": c.confidence,
    }


def _classification_from_dict(d: dict) -> ClassificationResult:
    return ClassificationResult(
        feedback_id=d["feedback_id"],
        category=Category(d["category"]),
        confidence=d["confidence"],
    )


def _sentiment_to_dict(s: SentimentResult) -> dict:
    return {
        "feedback_id": s.feedback_id,
        "sentiment": s.sentiment.value,
        "sentiment_score": s.sentiment_score,
        "urgency": s.urgency.value,
    }


def _sentiment_from_dict(d: dict) -> SentimentResult:
    return SentimentResult(
        feedback_id=d["feedback_id"],
        sentiment=Sentiment(d["sentiment"]),
        sentiment_score=d["sentiment_score"],
        urgency=Urgency(d["urgency"]),
    )


def _theme_tag_to_dict(t: ThemeTag) -> dict:
    return {"feedback_id": t.feedback_id, "theme": t.theme}


def _theme_tag_from_dict(d: dict) -> ThemeTag:
    return ThemeTag(feedback_id=d["feedback_id"], theme=d["theme"])


def _processed_feedback_to_dict(p: ProcessedFeedback) -> dict:
    return {
        "feedback": _feedback_item_to_dict(p.feedback),
        "classification": _classification_to_dict(p.classification),
        "sentiment": _sentiment_to_dict(p.sentiment),
        "themes": [_theme_tag_to_dict(t) for t in p.themes],
        "flagged_for_review": p.flagged_for_review,
        "review_reason": p.review_reason,
    }


def _processed_feedback_from_dict(d: dict) -> ProcessedFeedback:
    return ProcessedFeedback(
        feedback=_feedback_item_from_dict(d["feedback"]),
        classification=_classification_from_dict(d["classification"]),
        sentiment=_sentiment_from_dict(d["sentiment"]),
        themes=[_theme_tag_from_dict(t) for t in d.get("themes", [])],
        flagged_for_review=d.get("flagged_for_review", False),
        review_reason=d.get("review_reason"),
    )


# ---------------------------------------------------------------------------
# Public API — every other layer talks to storage through these functions
# ---------------------------------------------------------------------------

def save(processed: ProcessedFeedback, path: Path | None = None) -> None:
    """Append one processed feedback record to the store."""
    path = path if path is not None else DEFAULT_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_processed_feedback_to_dict(processed)) + "\n")


def save_many(processed_items: list[ProcessedFeedback], path: Path | None = None) -> None:
    """Append multiple records in one go (used by batch pipeline runs)."""
    path = path if path is not None else DEFAULT_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in processed_items:
            f.write(json.dumps(_processed_feedback_to_dict(item)) + "\n")


def load_all(path: Path | None = None) -> list[ProcessedFeedback]:
    """
    Load every record from the store.

    Returns an empty list — not an error — if the file doesn't exist yet.
    A brand-new project with no data processed is a normal state, not a
    failure state, so callers shouldn't have to wrap every call in a
    try/except just to handle "nothing's been saved yet."
    """
    path = path if path is not None else DEFAULT_STORE_PATH
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_processed_feedback_from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # One corrupted line (e.g. from a crash mid-write) should not
                # take down the whole read. Skip it, but don't fail silently —
                # surface exactly which line and why, so it's debuggable.
                print(f"[store] Skipping unreadable record at line {line_num}: {e}")
    return records


def load_by_id(feedback_id: str, path: Path | None = None) -> ProcessedFeedback | None:
    """Load a single record by its feedback id, or None if not found."""
    path = path if path is not None else DEFAULT_STORE_PATH
    for record in load_all(path):
        if record.feedback.id == feedback_id:
            return record
    return None


def clear(path: Path | None = None) -> None:
    """
    Wipe the store. Used by tests to guarantee a clean slate — never called
    from the main pipeline or API, since deleting all processed feedback is
    a destructive action that should never happen as a side effect of normal use.
    """
    path = path if path is not None else DEFAULT_STORE_PATH
    if os.path.exists(path):
        os.remove(path)