"""
Brain layer: sentiment and urgency scorer.

Same shape as classifier.py on purpose -- one focused responsibility,
same AIClient interface, same "never raises, always returns a safe
result" contract. Kept as a SEPARATE file from classifier.py rather than
combined into one giant "analyze everything" function, even though both
could technically be one API call -- see the module-level note below for
why that trade-off was made this way.
"""

from __future__ import annotations

import json

from brain.classifier import AIClient  # reuse the same interface, don't redefine it
from domain.feedback import FeedbackItem, Sentiment, SentimentResult, Urgency

# ---------------------------------------------------------------------------
# Design decision worth explaining to a mentor: separate call vs. combined call
# ---------------------------------------------------------------------------
# Classification and sentiment COULD be answered in a single API call (one
# prompt asking for category + sentiment + urgency all at once). We kept
# them as two separate functions/calls instead, because:
#   1. Each prompt stays focused and easy to reason about -- if sentiment
#      scoring starts misbehaving, you know exactly which prompt to fix,
#      without touching the classifier's few-shot examples at all.
#   2. It matches the layered-architecture rule: "one focused unit per
#      responsibility, not one giant function that does everything."
#   3. The cost is one extra API call per feedback item -- acceptable at
#      this project's scale, and worth trading for clarity.
# This is a deliberate trade-off, not an oversight -- be ready to explain
# it either way if asked "why not just one call?" (M5A2).


FEW_SHOT_EXAMPLES = [
    {
        "text": "The app crashes every time I try to upload a profile photo.",
        "sentiment": Sentiment.NEGATIVE,
        "sentiment_score": -0.7,
        "urgency": Urgency.HIGH,
    },
    {
        "text": "It would be great if you added a dark mode option.",
        "sentiment": Sentiment.NEUTRAL,
        "sentiment_score": 0.1,
        "urgency": Urgency.LOW,
    },
    {
        "text": "Love the redesign, it's so much cleaner than before!",
        "sentiment": Sentiment.POSITIVE,
        "sentiment_score": 0.9,
        "urgency": Urgency.LOW,
    },
    {
        "text": "I was charged twice for my subscription this month, please refund one.",
        "sentiment": Sentiment.NEGATIVE,
        "sentiment_score": -0.5,
        "urgency": Urgency.HIGH,
    },
    {
        # Deliberately picked: urgency is NOT the same axis as sentiment.
        # This is mildly negative in tone but nothing is broken or costing
        # the user money -- so urgency stays low. Without an example like
        # this, models tend to conflate "negative" with "urgent."
        "text": "The settings page could be laid out a bit better, took me a while to find dark mode.",
        "sentiment": Sentiment.NEGATIVE,
        "sentiment_score": -0.2,
        "urgency": Urgency.LOW,
    },
]

SYSTEM_PROMPT = """You are scoring customer feedback for sentiment and urgency.

Sentiment: Positive, Neutral, or Negative -- the emotional tone.
Sentiment score: a float from -1.0 (extremely negative) to 1.0 (extremely positive).
Urgency: Low, Medium, or High -- how quickly this needs human attention.
  - High: something is broken, costing the user money, or blocking their use of the product.
  - Medium: a real problem, but the user has a workaround or it's not blocking them.
  - Low: opinions, suggestions, minor friction, or praise.

Sentiment and urgency are DIFFERENT axes. Negative tone does not automatically mean
high urgency -- a mild complaint about layout is negative but low urgency; a billing
error is negative and high urgency.

Respond with ONLY a JSON object, no other text, no markdown fences:
{"sentiment": "<Positive|Neutral|Negative>", "sentiment_score": <float -1.0 to 1.0>, "urgency": "<Low|Medium|High>"}"""


def _build_few_shot_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(
            f'Feedback: "{ex["text"]}"\n'
            f'Output: {{"sentiment": "{ex["sentiment"].value}", '
            f'"sentiment_score": {ex["sentiment_score"]}, '
            f'"urgency": "{ex["urgency"].value}"}}'
        )
    return "\n\n".join(lines)


def _build_user_prompt(feedback_text: str) -> str:
    return f'{_build_few_shot_block()}\n\nFeedback: "{feedback_text}"\nOutput:'


def _parse_response(raw: str) -> tuple[Sentiment, float, Urgency]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    sentiment = Sentiment(data["sentiment"])
    score = float(data["sentiment_score"])
    urgency = Urgency(data["urgency"])
    return sentiment, score, urgency


def score_sentiment(feedback: FeedbackItem, client: AIClient) -> SentimentResult:
    """
    Score sentiment + urgency for one FeedbackItem. Same never-raises
    contract as classify_feedback: any failure falls back to a safe,
    honest, neutral/low result rather than crashing or guessing wildly.
    """
    if feedback.is_blank:
        # Neutral zero-score, low urgency: an empty submission is not
        # "bad" or "urgent," it's simply nothing to score.
        return SentimentResult(
            feedback_id=feedback.id,
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.0,
            urgency=Urgency.LOW,
        )

    try:
        raw = client.complete(
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(feedback.text),
            temperature=0.0,  # same reason as classifier.py: repeatable results
        )
        sentiment, score, urgency = _parse_response(raw)
        return SentimentResult(
            feedback_id=feedback.id,
            sentiment=sentiment,
            sentiment_score=score,
            urgency=urgency,
        )
    except Exception as e:
        print(f"[sentiment] Falling back to Neutral/Low for feedback_id={feedback.id}: {e}")
        return SentimentResult(
            feedback_id=feedback.id,
            sentiment=Sentiment.NEUTRAL,
            sentiment_score=0.0,
            urgency=Urgency.LOW,
        )