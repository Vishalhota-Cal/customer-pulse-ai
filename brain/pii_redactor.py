"""
Brain layer: PII redaction.

Pure regex-based pattern matching, no AI call -- deterministic, free, and
instant. This is a deliberate, honest baseline, not a claim of full
NLP-grade PII detection: it catches the common, structurally-obvious
patterns (emails, phone numbers, card-like number sequences) but will
miss anything that doesn't match a known shape (a name, a street address,
a PII detail phrased in unusual formatting). A production system at real
scale would pair this with an NLP-based tool like Microsoft Presidio for
broader coverage -- documented here as a known, deliberate limitation,
not hidden as a solved problem.

Called as early as possible in the pipeline -- before a FeedbackItem is
even constructed -- so redacted text is the only version that ever
reaches domain validation, persistence, or any AI API.
"""

from __future__ import annotations

import re

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Matches common phone formats: (123) 456-7890, 123-456-7890, 123.456.7890,
# +1 123 456 7890, etc. Deliberately requires at least 7 digits total to
# avoid false-positiving on short numeric sequences like "3 stars" or a
# ticket number.
_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Card-like sequences: 13-19 digits, optionally grouped with spaces or
# dashes (e.g. "4111 1111 1111 1111" or "4111-1111-1111-1111"). Anchored
# to end on a digit (not an optional trailing separator), so it doesn't
# swallow the space before the next word in the sentence.
_CARD_PATTERN = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list_of_pii_types_found).
    Never raises -- if somehow given non-string input, returns it unchanged
    with no types found, rather than crashing feedback ingestion over a
    privacy safeguard.
    """
    if not isinstance(text, str):
        return text, []

    found: list[str] = []
    redacted = text

    if _EMAIL_PATTERN.search(redacted):
        found.append("email")
        redacted = _EMAIL_PATTERN.sub("[EMAIL_REDACTED]", redacted)

    if _CARD_PATTERN.search(redacted):
        found.append("card_number")
        redacted = _CARD_PATTERN.sub("[CARD_REDACTED]", redacted)

    # Phone check runs AFTER card redaction so a long card-like number
    # doesn't also get double-matched and mangled by the phone pattern.
    if _PHONE_PATTERN.search(redacted):
        found.append("phone")
        redacted = _PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)

    return redacted, found