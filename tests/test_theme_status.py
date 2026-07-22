"""Tests for theme status persistence (persistence/store.py)."""

from pathlib import Path

import pytest

from persistence import store


@pytest.fixture
def status_path(tmp_path):
    return tmp_path / "theme_status_test.json"


def test_missing_file_returns_empty_dict(status_path):
    assert store.load_theme_statuses(status_path) == {}


def test_set_and_load_round_trips(status_path):
    store.set_theme_status("photo upload crash", "In Progress", status_path)
    assert store.load_theme_statuses(status_path) == {"photo upload crash": "In Progress"}


def test_multiple_themes_accumulate(status_path):
    store.set_theme_status("theme a", "Open", status_path)
    store.set_theme_status("theme b", "Resolved", status_path)
    result = store.load_theme_statuses(status_path)
    assert result == {"theme a": "Open", "theme b": "Resolved"}


def test_invalid_status_raises_value_error(status_path):
    with pytest.raises(ValueError):
        store.set_theme_status("theme a", "Not A Real Status", status_path)


def test_corrupted_file_returns_empty_dict_not_a_crash(status_path):
    status_path.write_text("not valid json{{{")
    assert store.load_theme_statuses(status_path) == {}


def test_updating_existing_theme_overwrites(status_path):
    store.set_theme_status("theme a", "Open", status_path)
    store.set_theme_status("theme a", "Resolved", status_path)
    assert store.load_theme_statuses(status_path) == {"theme a": "Resolved"}