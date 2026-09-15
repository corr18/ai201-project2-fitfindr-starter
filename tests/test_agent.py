"""
Tests for the FitFindr planning loop.

These cover the behaviour the loop is responsible for — branching on tool
results and passing state between tools — rather than the tools themselves.
Most tests stub the LLM-backed tools so the loop's decisions are tested
deterministically and without API calls.

Run from the repo root:
    python -m pytest tests/
"""

import agent
from agent import parse_query, run_agent
from tools import TOOL_ERROR
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe


# ── query parsing ─────────────────────────────────────────────────────────────

def test_parse_extracts_price_and_description():
    parsed = parse_query("looking for a vintage graphic tee under $30")
    assert parsed["description"] == "vintage graphic tee"
    assert parsed["max_price"] == 30.0
    assert parsed["size"] is None


def test_parse_extracts_size():
    assert parse_query("90s track jacket in size M")["size"] == "M"
    assert parse_query("black combat boots size 8")["size"] == "8"


def test_parse_extracts_all_three():
    parsed = parse_query("vintage graphic tee under $30, size M")
    assert parsed == {
        "description": "vintage graphic tee",
        "size": "M",
        "max_price": 30.0,
    }


def test_parse_does_not_mistake_words_for_sizes():
    """Lowercase letters inside words must not be read as a size."""
    assert parse_query("baggy jeans and sneakers")["size"] is None


# ── stubs ─────────────────────────────────────────────────────────────────────

def _stub_llm_tools(monkeypatch, outfit="Wear it with the baggy jeans.",
                    card="thrifted and thriving."):
    """Replace both LLM-backed tools so loop branching is tested offline."""
    monkeypatch.setattr(agent, "suggest_outfit", lambda item, wardrobe: outfit)
    monkeypatch.setattr(agent, "create_fit_card", lambda o, item: card)


# ── happy path and state ──────────────────────────────────────────────────────

def test_happy_path_populates_every_field(monkeypatch):
    _stub_llm_tools(monkeypatch)
    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())

    assert session["error"] is None
    assert session["selected_item"] is not None
    assert session["outfit_suggestion"] == "Wear it with the baggy jeans."
    assert session["fit_card"] == "thrifted and thriving."
    assert session["price_check"]["verdict"]


def test_state_passes_the_same_object_between_tools(monkeypatch):
    """
    The item search returned must be the *same object* handed to the next
    tools — not a copy, and not something re-derived from the query string.
    """
    seen = {}
    monkeypatch.setattr(
        agent, "suggest_outfit",
        lambda item, wardrobe: seen.setdefault("outfit_item", item) and "styled"
        or "styled",
    )
    monkeypatch.setattr(
        agent, "create_fit_card",
        lambda o, item: seen.setdefault("card_item", item) and "card" or "card",
    )

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())

    assert seen["outfit_item"] is session["selected_item"]
    assert seen["card_item"] is session["selected_item"]
    assert session["selected_item"] is session["search_results"][0]


def test_wardrobe_from_session_reaches_suggest_outfit(monkeypatch):
    wardrobe = get_example_wardrobe()
    captured = {}
    monkeypatch.setattr(
        agent, "suggest_outfit",
        lambda item, w: captured.setdefault("wardrobe", w) and "x" or "x",
    )
    monkeypatch.setattr(agent, "create_fit_card", lambda o, item: "card")

    run_agent("vintage graphic tee under $30", wardrobe)
    assert captured["wardrobe"] is wardrobe


def test_fit_card_receives_the_outfit_suggestion(monkeypatch):
    captured = {}
    monkeypatch.setattr(agent, "suggest_outfit", lambda i, w: "OUTFIT-TEXT")
    monkeypatch.setattr(
        agent, "create_fit_card",
        lambda o, item: captured.setdefault("outfit", o) and "card" or "card",
    )

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())
    assert captured["outfit"] == "OUTFIT-TEXT"
    assert session["outfit_suggestion"] == "OUTFIT-TEXT"


# ── branching ─────────────────────────────────────────────────────────────────

def test_no_results_stops_before_the_llm_tools(monkeypatch):
    """The core planning-loop requirement: downstream tools must not run."""
    called = []
    monkeypatch.setattr(
        agent, "suggest_outfit",
        lambda i, w: called.append("suggest_outfit") or "x",
    )
    monkeypatch.setattr(
        agent, "create_fit_card",
        lambda o, i: called.append("create_fit_card") or "x",
    )

    session = run_agent("designer ballgown size XXS under $5",
                        get_example_wardrobe())

    assert called == []
    assert session["error"]
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None


def test_no_results_error_names_the_constraints_it_used():
    session = run_agent("designer ballgown size XXS under $5",
                        get_example_wardrobe())
    assert "designer ballgown" in session["error"]
    assert "XXS" in session["error"]
    assert "$5" in session["error"]


def test_unparseable_query_calls_no_tools_at_all():
    session = run_agent("under $20", get_example_wardrobe())
    assert session["tool_calls"] == []
    assert session["error"]
    assert session["search_results"] == []


def test_agent_takes_different_paths_for_different_queries(monkeypatch):
    """Behaviour must change with the input — not a fixed 3-call pipeline."""
    _stub_llm_tools(monkeypatch)
    good = run_agent("vintage graphic tee under $30", get_example_wardrobe())
    bad = run_agent("designer ballgown size XXS under $5", get_example_wardrobe())
    none = run_agent("under $20", get_example_wardrobe())

    assert len(good["tool_calls"]) > len(bad["tool_calls"]) > len(none["tool_calls"])


# ── retry / fallback (stretch) ────────────────────────────────────────────────

def test_retry_drops_the_size_filter(monkeypatch):
    _stub_llm_tools(monkeypatch)
    session = run_agent("baggy carpenter jeans size S", get_example_wardrobe())

    assert session["retry"] is not None
    assert session["retry"]["dropped"] == "size"
    assert session["selected_item"] is not None
    assert any("size S" in note for note in session["notes"])


def test_retry_drops_the_price_cap(monkeypatch):
    _stub_llm_tools(monkeypatch)
    session = run_agent("corduroy wide-leg pants under $10", get_example_wardrobe())

    assert session["retry"] is not None
    assert "price" in session["retry"]["dropped"]
    assert session["selected_item"] is not None
    assert session["notes"]


def test_successful_first_search_does_not_retry(monkeypatch):
    _stub_llm_tools(monkeypatch)
    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())
    assert session["retry"] is None
    assert len([c for c in session["tool_calls"] if "search" in c]) == 1


# ── tool failures inside the loop ─────────────────────────────────────────────

def test_outfit_failure_keeps_the_listing_and_skips_the_card(monkeypatch):
    monkeypatch.setattr(
        agent, "suggest_outfit",
        lambda i, w: f"{TOOL_ERROR} styling model unreachable",
    )
    called = []
    monkeypatch.setattr(
        agent, "create_fit_card",
        lambda o, i: called.append("card") or "x",
    )

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())

    assert called == []
    assert session["error"]
    assert not session["error"].startswith(TOOL_ERROR)  # prefix stripped for the user
    assert session["selected_item"] is not None         # work so far is kept
    assert session["fit_card"] is None


def test_card_failure_keeps_the_outfit(monkeypatch):
    monkeypatch.setattr(agent, "suggest_outfit", lambda i, w: "a good outfit")
    monkeypatch.setattr(
        agent, "create_fit_card", lambda o, i: f"{TOOL_ERROR} caption model down",
    )

    session = run_agent("vintage graphic tee under $30", get_example_wardrobe())

    assert session["error"]
    assert session["outfit_suggestion"] == "a good outfit"  # not discarded
    assert session["fit_card"] is None


# ── empty wardrobe ────────────────────────────────────────────────────────────

def test_empty_wardrobe_still_completes_and_is_flagged(monkeypatch):
    _stub_llm_tools(monkeypatch, outfit="general advice")
    session = run_agent("vintage graphic tee under $30", get_empty_wardrobe())

    assert session["error"] is None
    assert session["fit_card"]
    assert any("general styling advice" in note for note in session["notes"])
