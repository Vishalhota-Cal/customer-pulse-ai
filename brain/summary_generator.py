"""
Brain layer: weekly narrative summary generator.

Takes a batch of ProcessedFeedback (everything classified, scored, and
theme-tagged this week) and produces a short, actionable narrative a
VP of CX could read and act on immediately (M5S5).

Temperature is set HIGHER here than in classifier.py/sentiment.py
(0.4 instead of 0.0) -- and that's a deliberate, different choice, not
an inconsistency. Classification and sentiment need to be repeatable:
same input, same category, every time. A narrative summary needs to
read like natural, fluent writing -- a small amount of variation in
phrasing doesn't hurt anything here the way it would hurt a
classification label.
"""

from __future__ import annotations

from collections import Counter

from brain.classifier import AIClient
from brain.theme_aggregator import aggregate_themes
from domain.feedback import ProcessedFeedback, Sentiment, Urgency

SYSTEM_PROMPT = """You write weekly customer feedback summaries for a VP of Customer Experience.

The summary must be 150-300 words and cover, in this order:
1. Overall sentiment trend this week (in plain language, not just percentages)
2. The top 2-3 recurring themes, named specifically
3. Any urgent items that need immediate attention (if there are none, say so plainly)
4. ONE clear, specific recommended action

Write in plain, direct prose. No headers, no bullet lists, no markdown --
this should read like a well-written paragraph a VP could act on
immediately without needing anything explained to them. Do not pad with
generic filler like "customers had mixed feelings" -- be specific and
grounded in the actual numbers and themes given to you."""


def _compute_stats(processed_items: list[ProcessedFeedback]) -> dict:
    """
    Turn a list of ProcessedFeedback into the plain aggregate numbers the
    summary prompt needs. Pure computation, no AI call -- deterministic
    and free, same reasoning as aggregate_themes() in theme_aggregator.py.
    """
    category_counts = Counter(p.classification.category.value for p in processed_items)
    sentiment_counts = Counter(p.sentiment.sentiment.value for p in processed_items)

    scores = [p.sentiment.sentiment_score for p in processed_items]
    avg_sentiment = sum(scores) / len(scores) if scores else 0.0

    urgent_items = [
        p for p in processed_items if p.sentiment.urgency == Urgency.HIGH
    ]

    all_theme_tags = [tag for p in processed_items for tag in p.themes]
    top_themes = aggregate_themes(all_theme_tags)[:5]  # top 5 is plenty for a prompt

    return {
        "total_items": len(processed_items),
        "category_counts": dict(category_counts),
        "sentiment_counts": dict(sentiment_counts),
        "avg_sentiment_score": round(avg_sentiment, 2),
        "urgent_count": len(urgent_items),
        "urgent_examples": [p.feedback.text[:150] for p in urgent_items[:3]],
        "top_themes": [(t.label, t.count) for t in top_themes],
    }


def _build_user_prompt(stats: dict) -> str:
    lines = [
        f"Total feedback items this week: {stats['total_items']}",
        f"Category breakdown: {stats['category_counts']}",
        f"Sentiment breakdown: {stats['sentiment_counts']}",
        f"Average sentiment score (-1.0 to 1.0): {stats['avg_sentiment_score']}",
        f"Number of HIGH urgency items: {stats['urgent_count']}",
    ]
    if stats["urgent_examples"]:
        lines.append("Example urgent items:")
        for ex in stats["urgent_examples"]:
            lines.append(f'  - "{ex}"')
    lines.append(f"Top recurring themes (theme, count): {stats['top_themes']}")
    lines.append("\nWrite the weekly summary now.")
    return "\n".join(lines)


def _fallback_summary(stats: dict) -> str:
    """
    Non-AI, templated fallback used only when the API call itself fails.
    This is deliberately boring and plain -- it exists so a VP still gets
    SOMETHING usable even during an outage, rather than nothing at all.
    It is clearly a template, not a claim of AI-generated insight.
    """
    themes_text = ", ".join(f"{label} ({count})" for label, count in stats["top_themes"]) or "none identified"
    return (
        f"[Automated fallback summary -- AI generation unavailable this run.] "
        f"Processed {stats['total_items']} feedback items this week. "
        f"Average sentiment score: {stats['avg_sentiment_score']} (-1.0 to 1.0 scale). "
        f"{stats['urgent_count']} item(s) flagged as high urgency and need review. "
        f"Top recurring themes: {themes_text}. "
        f"Recommended action: review flagged high-urgency items manually until AI "
        f"summary generation is restored."
    )


def generate_weekly_summary(
    processed_items: list[ProcessedFeedback], client: AIClient
) -> str:
    """
    Generate the weekly narrative summary. Never raises -- on API failure,
    falls back to a plain templated summary rather than crashing or
    returning nothing, since this is often the single artifact a
    stakeholder actually reads.
    """
    if not processed_items:
        # No feedback processed this week is a normal, expected state
        # (e.g. a slow week, or the very first run) -- not an error.
        return "No feedback was processed this week. Nothing to summarize yet."

    stats = _compute_stats(processed_items)

    try:
        summary = client.complete(
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(stats),
            temperature=0.4,
        )
        return summary.strip()
    except Exception as e:
        print(f"[summary_generator] Falling back to templated summary: {e}")
        return _fallback_summary(stats)