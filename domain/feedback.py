"""
Domain layer — plain data shapes only.

No AI calls, no persistence, no orchestration logic here. If you're tempted
to add a method that calls an API or writes to disk, it belongs in a
different layer. This file answers exactly one question: "what does a
piece of feedback, and everything derived from it, actually look like?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# Fixed taxonomy. Kept as an Enum (not a free string) so a typo like
# "Bugg" fails loudly at construction time instead of silently creating
# a new, never-matched category downstream in the aggregator.
class Category(str, Enum):
    BUG = "Bug / Technical Issue"
    BILLING = "Billing / Payments"
    FEATURE_REQUEST = "Feature Request"
    UX_COMPLAINT = "UX / Usability Complaint"
    CS_COMPLAINT = "Customer Service Complaint"
    PRAISE = "Positive Feedback / Praise"
    PERFORMANCE = "Performance / Speed"
    OTHER = "Other / Uncategorised"


class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class Urgency(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Hard cap on raw input length. This exists for two reasons at once:
# (1) security — an unbounded string is a cheap DoS vector against the
# API layer, and (2) cost/quality — a 50,000-character rant blows the
# model's effective context and produces a worse classification anyway.
# Enforced here, at construction time, so every layer above gets a
# FeedbackItem that is already guaranteed to be within bounds — nobody
# downstream has to remember to re-check it.
MAX_FEEDBACK_LENGTH = 5000


@dataclass(frozen=True)
class FeedbackItem:
    """A single raw piece of customer feedback, as submitted."""

    id: str
    text: str
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"  # e.g. "review", "ticket", "survey"

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("FeedbackItem.text must be a string")
        if len(self.text) > MAX_FEEDBACK_LENGTH:
            raise ValueError(
                f"FeedbackItem.text exceeds {MAX_FEEDBACK_LENGTH} characters "
                f"(got {len(self.text)}). Truncate or reject before construction."
            )

    @property
    def is_blank(self) -> bool:
        # Whitespace-only text ("   ", "\n\n") is functionally empty and
        # should be treated the same as "" by every downstream consumer —
        # this property is the single source of truth for that check, so
        # the classifier/sentiment layers don't each reinvent .strip() logic.
        return len(self.text.strip()) == 0


@dataclass(frozen=True)
class ClassificationResult:
    """Output of the classifier (brain layer) for one FeedbackItem."""

    feedback_id: str
    category: Category
    confidence: float  # 0.0-1.0

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass(frozen=True)
class SentimentResult:
    """Output of the sentiment/urgency scorer (brain layer) for one FeedbackItem."""

    feedback_id: str
    sentiment: Sentiment
    sentiment_score: float  # -1.0 (very negative) to +1.0 (very positive)
    urgency: Urgency

    def __post_init__(self):
        if not (-1.0 <= self.sentiment_score <= 1.0):
            raise ValueError(
                f"sentiment_score must be between -1.0 and 1.0, "
                f"got {self.sentiment_score}"
            )


@dataclass(frozen=True)
class ThemeTag:
    """A single specific theme extracted from one FeedbackItem, pre-clustering."""

    feedback_id: str
    theme: str  # e.g. "checkout button broken" — never a vague placeholder

    def __post_init__(self):
        if not self.theme.strip():
            raise ValueError("ThemeTag.theme cannot be blank")


@dataclass(frozen=True)
class AggregatedTheme:
    """A theme after cross-input clustering, with a frequency count."""

    label: str  # canonical name after de-duplication, e.g. "Checkout page issues"
    count: int
    example_feedback_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("AggregatedTheme.count must be >= 1")


@dataclass(frozen=True)
class ProcessedFeedback:
    """
    Everything known about one FeedbackItem after the full pipeline has
    run on it. This is what persistence stores and what the dashboard
    reads — the orchestration layer's job is to produce one of these
    per input item.
    """

    feedback: FeedbackItem
    classification: ClassificationResult
    sentiment: SentimentResult
    themes: list[ThemeTag] = field(default_factory=list)
    flagged_for_review: bool = False
    review_reason: str | None = None