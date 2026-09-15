"""
Tests for the FitFindr tools.

Run from the repo root with:
    python -m pytest tests/

The -m form puts the repo root on sys.path, which is what lets
`from tools import ...` resolve.

Tests that hit the Groq API are marked `llm` and skipped automatically when
GROQ_API_KEY is not set, so the deterministic tests still run in CI or on a
machine without a key:
    python -m pytest tests/ -m "not llm"
"""

import os

import pytest

from tools import (
    TOOL_ERROR,
    _size_matches,
    create_fit_card,
    estimate_price_fairness,
    search_listings,
    suggest_outfit,
)
from utils.data_loader import (
    get_empty_wardrobe,
    get_example_wardrobe,
    load_listings,
)

def needs_llm(func):
    """Mark a test as hitting the real API, and skip it when there's no key."""
    func = pytest.mark.llm(func)
    return pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set — skipping tests that call the model.",
    )(func)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_item():
    """A real listing from the dataset, used as tool input."""
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert results, "dataset should contain at least one graphic tee"
    return results[0]


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_empty_results():
    """The documented failure mode: an empty list, not an exception."""
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []


def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=40)
    assert all(item["price"] <= 40 for item in results)


def test_search_returns_full_listing_dicts():
    """The README documents these fields — downstream tools depend on them."""
    expected = {
        "id", "title", "description", "category", "style_tags", "size",
        "condition", "price", "colors", "brand", "platform",
    }
    for item in search_listings("denim jacket", size=None, max_price=None):
        assert expected <= set(item)


def test_search_sorted_by_relevance():
    """Results must be ordered best-first, since the agent takes index 0."""
    results = search_listings("vintage graphic tee", size=None, max_price=None)
    assert len(results) > 1
    top = results[0]
    top_text = f"{top['title']} {' '.join(top['style_tags'])}".lower()
    assert "tee" in top_text or "graphic" in top_text


def test_search_empty_description_returns_empty():
    assert search_listings("", size=None, max_price=None) == []
    assert search_listings("   ", size=None, max_price=None) == []


def test_search_size_filter_applied():
    results = search_listings("trousers", size="M", max_price=None)
    for item in results:
        assert _size_matches("M", item["size"])


@pytest.mark.parametrize(
    "wanted,listing_size,expected",
    [
        ("M", "M", True),
        ("M", "S/M", True),        # combined sizes must match
        ("M", "M/L", True),
        ("M", "Medium", True),     # alias expansion
        ("M", "XL (oversized)", False),
        ("S", "US 8", False),      # a naive substring test would match here
        ("8", "US 8", True),
        ("XS", "One Size", True),  # one size fits any request
    ],
)
def test_size_matching_rules(wanted, listing_size, expected):
    assert _size_matches(wanted, listing_size) is expected


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

@needs_llm
def test_suggest_outfit_with_wardrobe(sample_item):
    result = suggest_outfit(sample_item, get_example_wardrobe())
    assert isinstance(result, str)
    assert result.strip()
    assert not result.startswith(TOOL_ERROR)


@needs_llm
def test_suggest_outfit_empty_wardrobe_does_not_crash(sample_item):
    """The documented failure mode: general advice, not an exception."""
    result = suggest_outfit(sample_item, get_empty_wardrobe())
    assert isinstance(result, str)
    assert len(result.strip()) > 50
    assert not result.startswith(TOOL_ERROR)


def test_suggest_outfit_missing_item_returns_tool_error():
    """No item means no LLM call — returns a TOOL_ERROR string immediately."""
    assert suggest_outfit({}, get_example_wardrobe()).startswith(TOOL_ERROR)
    assert suggest_outfit(None, get_example_wardrobe()).startswith(TOOL_ERROR)


def test_suggest_outfit_llm_failure_returns_string(monkeypatch, sample_item):
    """A model failure becomes a TOOL_ERROR string, never a raised exception."""
    import tools

    def boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(tools, "_call_llm", boom)
    result = tools.suggest_outfit(sample_item, get_example_wardrobe())
    assert result.startswith(TOOL_ERROR)
    assert "simulated network failure" in result


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def test_fit_card_empty_outfit_returns_error_string(sample_item):
    """The documented failure mode: a descriptive string, not an exception."""
    result = create_fit_card("", sample_item)
    assert isinstance(result, str)
    assert result.startswith(TOOL_ERROR)


def test_fit_card_whitespace_outfit_returns_error_string(sample_item):
    assert create_fit_card("   \n  ", sample_item).startswith(TOOL_ERROR)


def test_fit_card_propagates_upstream_error(sample_item):
    """An upstream TOOL_ERROR is not fed to the model as if it were an outfit."""
    upstream = f"{TOOL_ERROR} styling model unreachable"
    assert create_fit_card(upstream, sample_item).startswith(TOOL_ERROR)


def test_fit_card_missing_item_returns_error_string():
    assert create_fit_card("wear it with jeans", {}).startswith(TOOL_ERROR)


@needs_llm
def test_fit_card_varies_between_calls(sample_item):
    """Different calls must not produce identical captions."""
    outfit = "Wear it with baggy jeans and chunky white sneakers, half tucked."
    first = create_fit_card(outfit, sample_item)
    second = create_fit_card(outfit, sample_item)
    assert not first.startswith(TOOL_ERROR)
    assert not second.startswith(TOOL_ERROR)
    assert first != second


@needs_llm
def test_fit_card_differs_for_different_items():
    tee = search_listings("graphic tee", None, 50)[0]
    jacket = search_listings("denim jacket", None, 80)[0]
    outfit = "Wear it with baggy jeans and chunky white sneakers."
    assert create_fit_card(outfit, tee) != create_fit_card(outfit, jacket)


# ── Tool 4: estimate_price_fairness (stretch) ─────────────────────────────────

def test_price_fairness_returns_expected_shape(sample_item):
    result = estimate_price_fairness(sample_item)
    assert set(result) == {
        "verdict", "item_price", "median_comparable",
        "comparable_count", "message",
    }
    assert result["verdict"] in {
        "great deal", "fair", "a bit high", "not enough data",
    }


def test_price_fairness_flags_an_overpriced_item():
    """A tee priced far above its comparables reads as 'a bit high'."""
    listings = load_listings()
    tee = dict(search_listings("graphic tee", None, 50)[0])
    tee["price"] = 500.0
    tee["id"] = "test_overpriced"
    assert estimate_price_fairness(tee, listings)["verdict"] == "a bit high"


def test_price_fairness_flags_a_bargain():
    listings = load_listings()
    tee = dict(search_listings("graphic tee", None, 50)[0])
    tee["price"] = 2.0
    tee["id"] = "test_bargain"
    assert estimate_price_fairness(tee, listings)["verdict"] == "great deal"


def test_price_fairness_handles_too_few_comparables():
    """The documented failure mode: a verdict, not a crash or a fake number."""
    lonely = {
        "id": "test_unique",
        "category": "spacesuits",
        "style_tags": ["nasa"],
        "price": 40.0,
    }
    result = estimate_price_fairness(lonely, load_listings())
    assert result["verdict"] == "not enough data"
    assert result["median_comparable"] is None
    assert result["comparable_count"] == 0


def test_price_fairness_excludes_the_item_itself(sample_item):
    """The item must not be its own comparable — that would skew the median."""
    result = estimate_price_fairness(sample_item)
    if result["comparable_count"]:
        assert result["comparable_count"] < len(load_listings())
