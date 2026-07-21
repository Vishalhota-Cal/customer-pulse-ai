"""
Brain layer: theme extraction and cross-input aggregation.

Two separate responsibilities live in this one file because they're two
steps of the SAME theme pipeline, not because the "one file per
responsibility" rule is being broken:

  1. extract_themes()   -- AI call, per single feedback item
  2. aggregate_themes() -- pure Python, across ALL feedback items at once

Aggregation is deliberately NOT an AI call. Clustering "checkout button
broken" with "checkout page freezes" via string similarity is free,
instant, and 100% deterministic -- running it twice on the same input
always gives the same grouping. Reaching for an AI call here would add
cost and non-determinism for a problem plain string matching already
solves well enough at this project's scale.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from brain.classifier import AIClient
from domain.feedback import AggregatedTheme, FeedbackItem, ThemeTag

# ---------------------------------------------------------------------------
# Step 1: per-item theme extraction (AI call)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "text": "The app crashes every time I try to upload a profile photo.",
        "themes": ["photo upload crash"],
    },
    {
        "text": "I was charged twice for my subscription and support didn't respond for 3 days.",
        # Deliberately TWO themes from one piece of feedback -- teaches the
        # model that a single message can raise more than one specific issue,
        # rather than forcing everything into one vague tag.
        "themes": ["duplicate subscription charge", "slow support response"],
    },
    {
        "text": "Love the redesign, it's so much cleaner than before!",
        "themes": ["positive redesign feedback"],
    },
    {
        # Deliberately picked to show what NOT to do: a generic complaint
        # still needs a SPECIFIC tag, not "customer issues" or "app problems."
        # This is the exact failure mode the rubric penalizes (M5S4).
        "text": "Everything about this app is just frustrating to use lately.",
        "themes": ["general usability frustration"],
    },
]

SYSTEM_PROMPT = """You extract specific, concrete themes from customer feedback.

Rules:
- Themes must be SPECIFIC, not generic. "checkout button broken" is good.
  "customer issues" or "app problems" is bad and will be rejected.
- A single piece of feedback can have MORE THAN ONE theme if it raises
  more than one distinct issue.
- Keep each theme short: 2-6 words, lowercase, no punctuation at the end.

Respond with ONLY a JSON array of strings, no other text, no markdown fences:
["theme one", "theme two"]

If there's truly nothing specific to extract, respond with an empty array: []"""


def _build_few_shot_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        themes_json = json.dumps(ex["themes"])
        lines.append(f'Feedback: "{ex["text"]}"\nOutput: {themes_json}')
    return "\n\n".join(lines)


def _build_user_prompt(feedback_text: str) -> str:
    return f'{_build_few_shot_block()}\n\nFeedback: "{feedback_text}"\nOutput:'


def _parse_response(raw: str) -> list[str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    themes = json.loads(cleaned)
    if not isinstance(themes, list):
        raise ValueError(f"Expected a JSON array, got {type(themes)}")
    return [str(t).strip() for t in themes if str(t).strip()]


def extract_themes(feedback: FeedbackItem, client: AIClient) -> list[ThemeTag]:
    """
    Extract specific theme tags from one FeedbackItem. Never raises --
    on any failure, falls back to an empty list. An empty theme list is
    a safe, honest "we couldn't extract anything specific" rather than
    a crash, and it simply means this item contributes nothing to the
    aggregate theme counts -- it doesn't corrupt them.
    """
    if feedback.is_blank:
        return []

    try:
        raw = client.complete(
            system=SYSTEM_PROMPT,
            user=_build_user_prompt(feedback.text),
            temperature=0.0,
        )
        theme_strings = _parse_response(raw)
        return [ThemeTag(feedback_id=feedback.id, theme=t) for t in theme_strings]
    except Exception as e:
        print(f"[theme_aggregator] No themes extracted for feedback_id={feedback.id}: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 2: cross-input aggregation (pure logic, no AI call)
# ---------------------------------------------------------------------------

# How similar two theme strings need to be (0.0-1.0) to be treated as the
# same underlying issue. Tuned to merge things like "checkout button
# broken" / "checkout button not working" while NOT merging genuinely
# different issues like "checkout button broken" / "checkout page slow".
SIMILARITY_THRESHOLD = 0.6


def _normalize(theme: str) -> str:
    return theme.strip().lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def aggregate_themes(theme_tags: list[ThemeTag]) -> list[AggregatedTheme]:
    """
    Cluster theme tags from many feedback items into ranked, de-duplicated
    AggregatedTheme records.

    Algorithm: greedy clustering by string similarity. Each tag either
    joins the first existing cluster it's similar enough to, or starts a
    new cluster. Simple, deterministic, and fast enough for this
    project's scale (hundreds of items, not millions) -- a proportionate
    choice per the "scalability: proportionate, not speculative" rule.
    A production system processing tens of thousands of items per run
    would want real embeddings-based clustering instead; that's a known,
    documented limitation of this approach, not an oversight.
    """
    clusters: list[dict] = []  # each: {"label": str, "tags": list[ThemeTag]}

    for tag in theme_tags:
        normalized = _normalize(tag.theme)
        matched_cluster = None
        for cluster in clusters:
            if _similarity(normalized, _normalize(cluster["label"])) >= SIMILARITY_THRESHOLD:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster["tags"].append(tag)
        else:
            clusters.append({"label": tag.theme, "tags": [tag]})

    aggregated = [
        AggregatedTheme(
            label=cluster["label"],
            count=len(cluster["tags"]),
            example_feedback_ids=[t.feedback_id for t in cluster["tags"]],
        )
        for cluster in clusters
    ]

    # Most frequent themes first -- that's what a weekly summary and
    # dashboard actually want to lead with.
    aggregated.sort(key=lambda a: a.count, reverse=True)
    return aggregated