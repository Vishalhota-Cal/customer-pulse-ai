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