"""
agent.py

The FitFindr planning loop. Orchestrates the tools in response to a natural
language user query, passing state between them via a session dict.

Usage:
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import re

from tools import (
    TOOL_ERROR,
    _tokenize,
    create_fit_card,
    estimate_price_fairness,
    search_listings,
    suggest_outfit,
)


# ── query parsing ─────────────────────────────────────────────────────────────

# "under $30", "below 30", "less than $30", "max $30", "$30 or less"
_PRICE_PATTERNS = [
    r"(?:under|below|less than|max|maximum|up to|no more than)\s*\$?\s*(\d+(?:\.\d+)?)",
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|or under|max)",
    r"\$\s*(\d+(?:\.\d+)?)",
]

# "size M", "size 8", "in a size 8", "size US 9"
_SIZE_PATTERN = r"\b(?:in\s+)?(?:a\s+)?size\s+(?:us\s+)?([a-z0-9/.]+)\b"

# Bare size tokens are only trusted when written in caps, so the "M" in
# "size M" is caught but the "s" in "jeans" is not.
_BARE_SIZES = {"XS", "S", "M", "L", "XL", "XXL"}

# Conversational lead-ins that add nothing to the search terms.
_LEAD_INS = [
    r"^\s*(?:i'?m\s+)?looking for\s+",
    r"^\s*i\s+(?:want|need)\s+",
    r"^\s*(?:can you\s+)?find me\s+",
    r"^\s*show me\s+",
    r"^\s*(?:a|an|some)\s+",
]


def parse_query(query: str) -> dict:
    """
    Extract search parameters from a natural language query.

    Uses regex rather than an LLM call: the three fields are shallow and
    regular, and keeping this step deterministic means a parse never costs an
    API call and never fails differently on two identical runs.

    Args:
        query: The raw user query.

    Returns:
        A dict with keys 'description' (str), 'size' (str | None) and
        'max_price' (float | None). 'description' is "" when the query says
        nothing about what the user wants — the caller treats that as a
        failure and stops before any tool runs.
    """
    text = (query or "").strip()
    remaining = text

    # --- price ---
    max_price = None
    for pattern in _PRICE_PATTERNS:
        match = re.search(pattern, remaining, flags=re.IGNORECASE)
        if match:
            max_price = float(match.group(1))
            remaining = remaining[: match.start()] + " " + remaining[match.end():]
            break

    # --- size ---
    size = None
    match = re.search(_SIZE_PATTERN, remaining, flags=re.IGNORECASE)
    if match:
        size = match.group(1).upper()
        remaining = remaining[: match.start()] + " " + remaining[match.end():]
    else:
        for token in re.findall(r"\b[A-Z]{1,3}\b", remaining):
            if token in _BARE_SIZES:
                size = token
                remaining = re.sub(rf"\b{token}\b", " ", remaining, count=1)
                break

    # --- description: whatever is left, minus filler ---
    description = re.sub(r"[,.;]+", " ", remaining)
    for lead_in in _LEAD_INS:
        description = re.sub(lead_in, "", description, flags=re.IGNORECASE)
    description = re.sub(r"\s+", " ", description).strip()

    return {"description": description, "size": size, "max_price": max_price}


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "price_check": None,         # dict from estimate_price_fairness
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "retry": None,               # set when the search was loosened
        "notes": [],                 # user-facing notes about what the agent did
        "tool_calls": [],            # ordered log of which tools actually ran
        "error": None,               # set if the interaction ended early
    }


def _log(session: dict, tool_name: str, outcome: str) -> None:
    """Record that a tool ran, so the loop's actual path is inspectable."""
    session["tool_calls"].append(f"{tool_name} -> {outcome}")


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    The loop is a sequence of guarded steps, each of which decides whether to
    run based on what the previous step returned. A run makes anywhere from
    zero tool calls (unparseable query) to four (search, price check, outfit,
    fit card); it is not a fixed pipeline.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and some output
        fields (outfit_suggestion, fit_card) may be None.
    """
    # Step 1: initialize state, then parse the query.
    session = _new_session(query, wardrobe)
    parsed = parse_query(query)
    session["parsed"] = parsed

    if not parsed["description"] or not _tokenize(parsed["description"]):
        # Branch: nothing searchable left after filler words are removed.
        # Return before calling any tool at all.
        session["error"] = (
            "I couldn't tell what you're looking for from that. Try naming the "
            "piece itself — for example \"vintage graphic tee under $30, size M\"."
        )
        return session

    # Step 2: search, with two levels of loosening if the first attempt is empty.
    description = parsed["description"]
    size = parsed["size"]
    max_price = parsed["max_price"]

    results = search_listings(description, size, max_price)
    _log(session, "search_listings", f"{len(results)} results")

    if not results and size:
        # Retry without the size filter — size is the constraint most likely to
        # be the sole reason a real match was excluded.
        results = search_listings(description, None, max_price)
        _log(session, "search_listings (retry, no size)", f"{len(results)} results")
        if results:
            session["retry"] = {"dropped": "size", "original_size": size}
            session["notes"].append(
                f"No listings in size {size}, so I searched every size instead."
            )

    if not results and max_price is not None:
        # Retry once more with the price cap dropped as well.
        results = search_listings(description, None, None)
        _log(session, "search_listings (retry, no size/price)", f"{len(results)} results")
        if results:
            session["retry"] = {
                "dropped": "size and price" if size else "price",
                "original_size": size,
                "original_max_price": max_price,
            }
            dropped_phrase = "those filters" if size else "the price cap"
            session["notes"].append(
                f"Nothing matched under ${max_price:.0f}"
                + (f" in size {size}" if size else "")
                + f", so I dropped {dropped_phrase} — the closest match is "
                "above your budget."
            )

    if not results:
        # Branch: give up here. suggest_outfit is never called on empty input.
        constraints = [f"\"{description}\""]
        if size:
            constraints.append(f"size {size}")
        if max_price is not None:
            constraints.append(f"under ${max_price:.0f}")
        session["error"] = (
            f"No listings matched {', '.join(constraints)}. "
            + (
                "I also retried without the size and price filters and still found "
                "nothing, so it's the description that's the problem. "
                if (size or max_price is not None)
                else ""
            )
            + "Try different words for the piece (the dataset is tops, bottoms, "
            "outerwear, shoes and accessories, $12–$75), or search for something "
            "adjacent — \"denim jacket\", \"graphic tee\", \"wide-leg trousers\"."
        )
        return session

    session["search_results"] = results

    # Step 3: select the best match. This exact dict object is what every
    # downstream tool receives — nothing is re-derived from the query string.
    session["selected_item"] = results[0]

    # Step 4 (stretch): price check. Advisory only — never ends the run.
    price_check = estimate_price_fairness(session["selected_item"])
    session["price_check"] = price_check
    _log(session, "estimate_price_fairness", price_check["verdict"])

    # Step 5: style it, using the item from step 3 and the session's wardrobe.
    wardrobe_is_empty = not (session["wardrobe"] or {}).get("items")
    outfit = suggest_outfit(session["selected_item"], session["wardrobe"])

    if outfit.startswith(TOOL_ERROR):
        # Branch: styling failed. Keep the listing we found, skip the fit card.
        _log(session, "suggest_outfit", "failed")
        session["error"] = outfit.replace(TOOL_ERROR, "").strip()
        return session

    _log(session, "suggest_outfit", "ok")
    session["outfit_suggestion"] = outfit
    if wardrobe_is_empty:
        session["notes"].append(
            "This is general styling advice — you haven't added a wardrobe yet, "
            "so I couldn't name pieces you already own."
        )

    # Step 6: caption it, using the step 5 output and the step 3 item.
    fit_card = create_fit_card(session["outfit_suggestion"], session["selected_item"])

    if fit_card.startswith(TOOL_ERROR):
        # Branch: caption failed, but the listing and outfit above are still
        # good — surface the error without discarding work that succeeded.
        _log(session, "create_fit_card", "failed")
        session["error"] = fit_card.replace(TOOL_ERROR, "").strip()
        return session

    _log(session, "create_fit_card", "ok")
    session["fit_card"] = fit_card

    # Step 7: done — every downstream step succeeded.
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Parsed: {session['parsed']}")
        print(f"Found: {session['selected_item']['title']}")
        print(f"Price check: {session['price_check']['message']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

        # State passing, proved by identity rather than equality: the dict the
        # search returned IS the dict the later tools were handed.
        assert session["selected_item"] is session["search_results"][0]
        print("\n[state] selected_item is search_results[0]:", True)

    print(f"\nTools called: {session['tool_calls']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
    print(f"Tools called: {session2['tool_calls']}")
    print(f"fit_card is None: {session2['fit_card'] is None}")
    print(f"outfit_suggestion is None: {session2['outfit_suggestion'] is None}")

    print("\n\n=== Retry path: size filter loosened ===\n")
    session3 = run_agent(
        query="baggy carpenter jeans size S",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Parsed: {session3['parsed']}")
    print(f"Retry: {session3['retry']}")
    print(f"Notes: {session3['notes']}")
    print(f"Tools called: {session3['tool_calls']}")

    print("\n\n=== Retry path: price cap loosened ===\n")
    session5 = run_agent(
        query="corduroy wide-leg pants under $10",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Parsed: {session5['parsed']}")
    print(f"Retry: {session5['retry']}")
    print(f"Notes: {session5['notes']}")
    print(f"Tools called: {session5['tool_calls']}")

    print("\n\n=== Empty wardrobe path ===\n")
    session4 = run_agent(
        query="vintage denim jacket under $60",
        wardrobe=get_empty_wardrobe(),
    )
    print(f"Notes: {session4['notes']}")
    print(f"Outfit: {(session4['outfit_suggestion'] or '')[:200]}")
