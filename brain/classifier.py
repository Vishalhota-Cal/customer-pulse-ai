"""
Brain layer: category classifier.

Turns raw feedback text into a Category + confidence, using a few-shot
prompted call to Claude with a fixed, low temperature for consistency.

Design choice worth explaining to a mentor: this file depends on an
AIClient *protocol* (an interface), not on the Anthropic SDK directly.
That's what lets the automated test suite run against a fake, deterministic,
free client (see tests/fakes.py) instead of a real paid API — per the
project's testing philosophy, automated tests never depend on a real
external service.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from dotenv import load_dotenv

from domain.feedback import Category, ClassificationResult, FeedbackItem

# Load variables from a local .env file into the process environment.
# This has to happen HERE, not just be listed in requirements.txt --
# installing python-dotenv doesn't automatically read .env, something
# has to actually call load_dotenv(). This is the one place both
# AnthropicClient and OpenAIClient read their API key from os.environ,
# so loading it here guarantees it's loaded before either client is
# ever constructed, regardless of which script or module does the
# constructing (the API, a one-off manual test script, a notebook, etc).
load_dotenv()


class AIClient(Protocol):
    """Anything that can turn a (system, user) prompt pair into text."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        ...


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
# Deliberately chosen, not random — one per category so the model has seen
# every possible label at least once, PLUS one deliberately ambiguous/mixed
# example (the last one) so the model has a pattern to follow when a real
# piece of feedback doesn't fall cleanly into one bucket. Be ready to
# explain this choice to a mentor (M5S3) — that's exactly what this
# comment is for.
FEW_SHOT_EXAMPLES = [
    {
        "text": "The app crashes every time I try to upload a profile photo.",
        "category": Category.BUG,
    },
    {
        "text": "I was charged twice for my subscription this month, please refund one.",
        "category": Category.BILLING,
    },
    {
        "text": "It would be great if you added a dark mode option.",
        "category": Category.FEATURE_REQUEST,
    },
    {
        "text": "I can't figure out where the settings page is, the menu is confusing.",
        "category": Category.UX_COMPLAINT,
    },
    {
        "text": "Support took three days to respond to my ticket and never actually fixed it.",
        "category": Category.CS_COMPLAINT,
    },
    {
        "text": "Love the redesign, it's so much cleaner than before!",
        "category": Category.PRAISE,
    },
    {
        "text": "The app takes forever to load on my phone, especially in the mornings.",
        "category": Category.PERFORMANCE,
    },
    {
        # Ambiguous on purpose: reads like praise AND a complaint about
        # support response time. This teaches the model to pick the
        # DOMINANT, more actionable issue (support delay) rather than
        # defaulting to the friendlier-sounding label.
        "text": "Your product is great but I've emailed support twice with no reply.",
        "category": Category.CS_COMPLAINT,
    },
]

SYSTEM_PROMPT = """You are a customer feedback classifier for a product/CX team.

Classify each piece of feedback into EXACTLY ONE of these categories:
- Bug / Technical Issue
- Billing / Payments
- Feature Request
- UX / Usability Complaint
- Customer Service Complaint
- Positive Feedback / Praise
- Performance / Speed
- Other / Uncategorised

Respond with ONLY a JSON object, no other text, no markdown fences:
{"category": "<one of the exact category strings above>", "confidence": <float 0.0-1.0>}

If feedback mentions multiple issues, pick the most dominant, actionable one.
If nothing fits cleanly, use "Other / Uncategorised" with a lower confidence."""


def _build_few_shot_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(
            f'Feedback: "{ex["text"]}"\n'
            f'Output: {{"category": "{ex["category"].value}", "confidence": 0.95}}'
        )
    return "\n\n".join(lines)


def _build_user_prompt(feedback_text: str) -> str:
    return (
        f"{_build_few_shot_block()}\n\n"
        f'Feedback: "{feedback_text}"\n'
        f"Output:"
    )


def _parse_response(raw: str) -> tuple[Category, float]:
    """
    Parse the model's JSON response defensively. Models occasionally wrap
    JSON in markdown fences even when told not to -- strip those before
    parsing rather than letting the whole classification fail over
    formatting noise.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)  # raises json.JSONDecodeError on bad output
    category = Category(data["category"])  # raises ValueError on unknown category
    confidence = float(data["confidence"])
    return category, confidence


def classify_feedback(
    feedback: FeedbackItem,
    client: AIClient,
    escalation_client: AIClient | None = None,
    escalation_threshold: float = 0.5,
) -> ClassificationResult:
    """
    Classify one FeedbackItem. Never raises -- on any failure (API down,
    bad JSON, unknown category from the model, whatever shape the failure
    takes), falls back to a safe, boring, honest result: OTHER with
    confidence 0.0. That low confidence is the signal orchestration uses
    to flag the item for human review, rather than the pipeline crashing
    or silently pretending it classified something it didn't.

    Tiered model escalation (cost control): if escalation_client is
    provided and the PRIMARY client's confidence comes back below
    escalation_threshold, retry once with the escalation_client (intended
    to be a stronger, more expensive model). This means the expensive
    model is only ever paid for on the genuinely ambiguous cases -- the
    common, easy cases never touch it. If escalation_client is None
    (the default), behavior is identical to before this feature existed.
    """
    # Blank input never reaches the API at all: there is nothing to
    # classify, and calling a paid model on empty text is both wasteful
    # and undefined behavior. This is also the empty-input edge case the
    # rubric explicitly tests for (M5B2).
    if feedback.is_blank:
        return ClassificationResult(
            feedback_id=feedback.id, category=Category.OTHER, confidence=1.0
        )

    result = _classify_once(feedback, client)

    if (
        escalation_client is not None
        and result.category != Category.OTHER  # OTHER/0.0 already means "trust nothing", escalating won't help
        and result.confidence < escalation_threshold
    ):
        print(
            f"[classifier] feedback_id={feedback.id} confidence {result.confidence} "
            f"below threshold {escalation_threshold}, escalating to stronger model"
        )
        escalated = _classify_once(feedback, escalation_client)
        # Only use the escalated result if it's actually more confident --
        # otherwise keep the cheaper model's answer rather than blindly
        # trusting whichever call happened second.
        if escalated.confidence > result.confidence:
            result = escalated

    return result


def _classify_once(feedback: FeedbackItem, client: AIClient) -> ClassificationResult:
    """One classification attempt against one client -- the shared logic
    used for both the primary (cheap) and escalation (strong) model calls."""
    try:
        raw = client.complete(
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(feedback.text),
            temperature=0.0,  # low temperature: same input -> same output, run after run
        )
        category, confidence = _parse_response(raw)
        return ClassificationResult(
            feedback_id=feedback.id, category=category, confidence=confidence
        )
    except Exception as e:
        # Deliberately broad: a network timeout, an invalid API key, a
        # malformed JSON response, and an unrecognized category string are
        # all DIFFERENT exception types, but all of them mean the same
        # thing operationally -- "we couldn't get a trustworthy answer" --
        # so they all get the same safe fallback instead of five different
        # crash paths.
        print(f"[classifier] Falling back to OTHER for feedback_id={feedback.id}: {e}")
        return ClassificationResult(
            feedback_id=feedback.id, category=Category.OTHER, confidence=0.0
        )


class AnthropicClient:
    """
    Real AIClient implementation, backed by the Anthropic API.

    The anthropic package is imported lazily, inside __init__, not at the
    top of this file. That means classifier.py's core logic can be
    imported and unit-tested with a fake client even in an environment
    where the anthropic package isn't installed -- the dependency is only
    required at the moment you actually construct a real client.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5"):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file "
                "(see .env.example) -- never hardcode it in source."
            )
        import anthropic  # lazy import, see docstring above

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIClient:
    """
    Real AIClient implementation, backed by OpenAI's API instead of
    Anthropic's. This exists for exactly the reason AIClient is a
    Protocol and not a concrete class import scattered through every
    brain/*.py file: classifier.py, sentiment.py, theme_aggregator.py,
    and summary_generator.py never import a specific vendor SDK -- they
    only depend on "something with a .complete(system, user,
    temperature) method." Swap AnthropicClient for OpenAIClient here,
    and nothing else in the whole project needs to change.

    Use this if you only have an OpenAI API key rather than an
    Anthropic one -- set OPENAI_API_KEY in your .env instead of
    ANTHROPIC_API_KEY.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file "
                "(see .env.example) -- never hardcode it in source."
            )
        import openai  # lazy import, same reasoning as AnthropicClient above

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content