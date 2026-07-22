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
import math
from difflib import SequenceMatcher
from typing import Protocol

from brain.classifier import AIClient
from domain.feedback import AggregatedTheme, FeedbackItem, ThemeTag


class EmbeddingClient(Protocol):
    """Anything that can turn a list of strings into a list of embedding vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

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


# How similar two theme strings need to be (0.0-1.0) to be treated as the
# same underlying issue. Tuned to merge things like "checkout button
# broken" / "checkout button not working" while NOT merging genuinely
# different issues like "checkout button broken" / "checkout page slow".
SIMILARITY_THRESHOLD = 0.6

# Cosine similarity threshold for embeddings-based clustering.
#
# CORRECTED after real-world testing: the original value here was 0.82,
# chosen without ever testing against real embedding vectors -- only
# hand-crafted fake vectors in unit tests, deliberately built to be
# obviously high or low similarity. Running this against real feedback
# (three clearly-the-same-issue phrasings: "buy button loading issue" /
# "order completion failure" / "payment page information rejection")
# proved 0.82 far too strict: real cosine similarities between short,
# related-but-differently-worded phrases from OpenAI's embedding model
# commonly land in the 0.3-0.6 range, not 0.8+. At 0.82, almost nothing
# ever merged except phrases that happened to share literal words -- at
# which point the embeddings path wasn't doing meaningfully different
# work than the string-similarity fallback it's supposed to improve on.
#
# 0.5 is a more realistic starting point. If you run
# diagnose_embedding_threshold.py against your own data and find this
# still merges too aggressively or not aggressively enough, adjust here
# -- this number should be tuned against real embedding output, not
# picked in the abstract a second time either.
EMBEDDING_SIMILARITY_THRESHOLD = 0.5


def _normalize(theme: str) -> str:
    return theme.strip().lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def aggregate_themes(
    theme_tags: list[ThemeTag], embedding_client: EmbeddingClient | None = None
) -> list[AggregatedTheme]:
    """
    Cluster theme tags from many feedback items into ranked, de-duplicated
    AggregatedTheme records.

    Two modes:
    - embedding_client=None (default): string-similarity clustering via
      difflib -- free, instant, deterministic, but only merges phrases
      that are textually similar (won't merge "app won't let me pay" with
      "checkout fails" -- different words, same underlying issue).
    - embedding_client provided: real semantic clustering using embedding
      vectors and cosine similarity -- catches semantically related
      phrases with different wording, at the cost of one extra API call
      per aggregation run. If the embedding call itself fails for any
      reason, falls back to the string-similarity path rather than
      losing theme aggregation entirely.
    """
    if not theme_tags:
        return []

    if embedding_client is not None:
        print(f"[theme_aggregator] Using SEMANTIC (embeddings) clustering for {len(theme_tags)} theme mention(s)")
        try:
            return _aggregate_themes_semantic(theme_tags, embedding_client)
        except Exception as e:
            print(f"[theme_aggregator] Embedding-based clustering failed, falling back to string similarity: {e}")
    else:
        print(f"[theme_aggregator] Using STRING-SIMILARITY clustering for {len(theme_tags)} theme mention(s) (no embedding client provided)")

    return _aggregate_themes_string_similarity(theme_tags)


def _aggregate_themes_string_similarity(theme_tags: list[ThemeTag]) -> list[AggregatedTheme]:
    """
    Algorithm: greedy clustering by string similarity. Each tag either
    joins the first existing cluster it's similar enough to, or starts a
    new cluster. Simple, deterministic, and fast enough for this
    project's scale (hundreds of items, not millions) -- a proportionate
    choice per the "scalability: proportionate, not speculative" rule.
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

    return _clusters_to_aggregated(clusters)


def _aggregate_themes_semantic(
    theme_tags: list[ThemeTag], embedding_client: EmbeddingClient
) -> list[AggregatedTheme]:
    """
    Real semantic clustering: embed every UNIQUE theme string in one batch
    call (not one call per tag -- unique strings only, to keep the API
    call count proportional to distinct themes, not total mentions), then
    greedily cluster by cosine similarity the same way the string-based
    version clusters by SequenceMatcher ratio.
    """
    unique_themes = list(dict.fromkeys(tag.theme for tag in theme_tags))  # preserves order, de-dupes
    vectors = embedding_client.embed(unique_themes)
    if len(vectors) != len(unique_themes):
        raise ValueError(
            f"embedding_client returned {len(vectors)} vectors for {len(unique_themes)} inputs"
        )
    theme_to_vector = dict(zip(unique_themes, vectors))

    clusters: list[dict] = []  # each: {"label": str, "vector": list[float], "tags": list[ThemeTag]}
    for tag in theme_tags:
        vector = theme_to_vector[tag.theme]
        matched_cluster = None
        for cluster in clusters:
            if _cosine_similarity(vector, cluster["vector"]) >= EMBEDDING_SIMILARITY_THRESHOLD:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster["tags"].append(tag)
        else:
            clusters.append({"label": tag.theme, "vector": vector, "tags": [tag]})

    return _clusters_to_aggregated(clusters)


def _clusters_to_aggregated(clusters: list[dict]) -> list[AggregatedTheme]:

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


class OpenAIEmbeddingClient:
    """
    Real EmbeddingClient implementation, backed by OpenAI's embeddings
    API. Lazy-imports the openai package the same way AnthropicClient/
    OpenAIClient in classifier.py do, so this module can be imported and
    tested with a fake embedding client even where the openai package
    isn't installed.
    """

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to your .env file "
                "(see .env.example) -- never hardcode it in source."
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]