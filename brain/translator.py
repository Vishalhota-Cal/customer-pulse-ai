"""
Brain layer: translation for non-English feedback.

Replaces the old "detect non-English, flag it, never look at it again"
behavior with "detect non-English, translate it, THEN run the normal
pipeline on it" -- so non-English feedback actually gets classified,
scored, and theme-tagged instead of being permanently skipped.

Still flagged for human review afterward -- not because it was skipped,
but because a translation step adds a real, honest source of extra
uncertainty (translation quality) that a human may want to spot-check.
"""

from __future__ import annotations

from brain.classifier import AIClient

SYSTEM_PROMPT = """You are a translator. Translate the given text to English.

Respond with ONLY the translated text -- no commentary, no notes, no
markdown fences, no quotation marks around the translation."""


def translate_to_english(text: str, client: AIClient) -> str | None:
    """
    Translate text to English. Returns None on any failure (API outage,
    empty response) -- callers are expected to fall back to the old
    flag-and-skip behavior when this returns None, rather than passing
    a None or garbled string into the classification pipeline.
    """
    try:
        result = client.complete(
            system=SYSTEM_PROMPT,
            user=text,
            temperature=0.0,  # consistency: same input, same translation, every run
        )
        result = result.strip()
        return result if result else None
    except Exception as e:
        print(f"[translator] Translation failed: {e}")
        return None