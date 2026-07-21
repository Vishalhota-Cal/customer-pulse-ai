"""
Shared fake AI client for the automated test suite.

This is THE reason the test suite never needs a real Anthropic API key,
never costs money to run, and never gives a different answer on two
runs of the same test. Per the project's testing philosophy: automated
tests run against a fake/mock version of any external dependency.
"""

from __future__ import annotations


class ScriptedAIClient:
    """
    A fake AIClient that returns a pre-scripted response based on which
    kind of prompt it's being asked (classification vs sentiment vs
    theme extraction), detected by looking at the system prompt content.
    This mirrors how the real classifier/sentiment/theme_aggregator
    modules each send a distinctly-worded system prompt.
    """

    def __init__(
        self,
        classification_response: str = '{"category": "Other / Uncategorised", "confidence": 0.5}',
        sentiment_response: str = '{"sentiment": "Neutral", "sentiment_score": 0.0, "urgency": "Low"}',
        theme_response: str = "[]",
        summary_response: str = "Fake weekly summary.",
    ):
        self.classification_response = classification_response
        self.sentiment_response = sentiment_response
        self.theme_response = theme_response
        self.summary_response = summary_response
        self.call_count = 0
        self.last_system = None
        self.last_user = None

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        self.last_system = system
        self.last_user = user

        system_lower = system.lower()
        if "urgency" in system_lower:
            return self.sentiment_response
        if "category" in system_lower:
            return self.classification_response
        if "weekly customer feedback" in system_lower:
            return self.summary_response
        return self.theme_response


class ExplodingAIClient:
    """A fake AIClient that always raises -- used to test failure paths."""

    def __init__(self, exception: Exception | None = None):
        self.exception = exception or ConnectionError("simulated API outage")

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise self.exception