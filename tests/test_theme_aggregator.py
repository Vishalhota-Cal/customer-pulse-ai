"""Tests for brain/theme_aggregator.py -- extraction (fake client) and aggregation (pure logic)."""

from domain.feedback import FeedbackItem, ThemeTag
from brain.theme_aggregator import aggregate_themes, extract_themes
from tests.fakes import ExplodingAIClient, ScriptedAIClient


def test_blank_input_returns_empty_list_without_calling_api():
    fake = ScriptedAIClient()
    result = extract_themes(FeedbackItem(id="fb0", text=""), fake)
    assert result == []
    assert fake.call_count == 0


def test_single_theme_extracted():
    fake = ScriptedAIClient(theme_response='["photo upload crash"]')
    result = extract_themes(FeedbackItem(id="fb1", text="crashes on upload"), fake)
    assert len(result) == 1
    assert result[0].theme == "photo upload crash"


def test_multiple_themes_from_one_item():
    fake = ScriptedAIClient(theme_response='["duplicate charge", "slow support response"]')
    result = extract_themes(FeedbackItem(id="fb2", text="charged twice, support ignored me"), fake)
    assert len(result) == 2


def test_malformed_response_falls_back_to_empty_list():
    fake = ScriptedAIClient(theme_response="not a json array")
    result = extract_themes(FeedbackItem(id="fb3", text="whatever"), fake)
    assert result == []


def test_api_outage_does_not_crash():
    result = extract_themes(FeedbackItem(id="fb4", text="does this crash?"), ExplodingAIClient())
    assert result == []


def test_near_duplicate_themes_cluster_together():
    tags = [
        ThemeTag(feedback_id="a", theme="checkout button broken"),
        ThemeTag(feedback_id="b", theme="checkout button not working"),
        ThemeTag(feedback_id="c", theme="checkout button broken"),
        ThemeTag(feedback_id="d", theme="app is slow to load"),
    ]
    aggregated = aggregate_themes(tags)
    checkout_related = [a for a in aggregated if "checkout" in a.label.lower()]
    assert sum(a.count for a in checkout_related) == 3


def test_distinct_themes_stay_separate():
    tags = [
        ThemeTag(feedback_id="a", theme="dark mode request"),
        ThemeTag(feedback_id="b", theme="checkout button broken"),
    ]
    aggregated = aggregate_themes(tags)
    assert len(aggregated) == 2


def test_aggregation_sorted_by_count_descending():
    tags = [ThemeTag(feedback_id=str(i), theme="common issue") for i in range(3)]
    tags.append(ThemeTag(feedback_id="x", theme="rare issue"))
    aggregated = aggregate_themes(tags)
    assert aggregated[0].count >= aggregated[-1].count


def test_empty_theme_list_returns_empty_result():
    assert aggregate_themes([]) == []


class _FakeEmbeddingClient:
    def __init__(self, vector_map, should_raise=False):
        self.vector_map = vector_map
        self.should_raise = should_raise
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        if self.should_raise:
            raise ConnectionError("simulated embedding API outage")
        return [self.vector_map[t] for t in texts]


def test_semantic_clustering_merges_differently_worded_similar_themes():
    tags = [
        ThemeTag(feedback_id="a", theme="app wont let me pay"),
        ThemeTag(feedback_id="b", theme="checkout fails every time"),
        ThemeTag(feedback_id="c", theme="dark mode request"),
    ]
    fake = _FakeEmbeddingClient({
        "app wont let me pay": [1.0, 0.02, 0.0, 0.0],
        "checkout fails every time": [0.98, 0.05, 0.0, 0.0],
        "dark mode request": [0.0, 0.0, 1.0, 0.02],
    })
    result = aggregate_themes(tags, embedding_client=fake)
    assert fake.call_count == 1  # one batched call, not one per tag
    merged = [r for r in result if r.count == 2]
    assert len(merged) == 1
    assert len(result) == 2


def test_semantic_clustering_falls_back_on_embedding_failure():
    tags = [ThemeTag(feedback_id="a", theme="checkout page freeze"), ThemeTag(feedback_id="b", theme="dark mode request")]
    fake = _FakeEmbeddingClient({}, should_raise=True)
    result = aggregate_themes(tags, embedding_client=fake)
    assert len(result) == 2  # falls back to string similarity, doesn't crash -- these two are genuinely dissimilar strings


def test_no_embedding_client_uses_default_string_similarity():
    tags = [
        ThemeTag(feedback_id="a", theme="checkout button broken"),
        ThemeTag(feedback_id="b", theme="checkout button not working"),
    ]
    result = aggregate_themes(tags)  # no embedding_client passed
    assert len(result) == 1
    assert result[0].count == 2


def test_threshold_merges_moderate_similarity_but_not_low_similarity():
    """
    Regression test for the threshold correction (0.82 -> 0.5): real
    embeddings of related-but-differently-worded short phrases commonly
    land around 0.5-0.6 cosine similarity, not 0.8+. This checks the
    threshold actually merges at a REALISTIC similarity level, not just
    the extreme near-1.0/near-0.0 vectors the other tests use.
    """
    tags = [
        ThemeTag(feedback_id="a", theme="order completion failure"),
        ThemeTag(feedback_id="b", theme="payment page rejection"),
        ThemeTag(feedback_id="c", theme="totally unrelated topic"),
    ]
    fake = _FakeEmbeddingClient({
        "order completion failure": [1.0, 0.0, 0.0],
        "payment page rejection": [0.55, 0.83, 0.0],   # cosine similarity ~0.55 with the first
        "totally unrelated topic": [0.0, 0.1, 1.0],      # cosine similarity ~0.0 with the first
    })
    result = aggregate_themes(tags, embedding_client=fake)
    merged = [r for r in result if r.count == 2]
    assert len(merged) == 1  # first two merge at ~0.55 similarity
    assert len(result) == 2  # third stays separate at ~0.0 similarity