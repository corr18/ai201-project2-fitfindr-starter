"""
tools.py

The FitFindr tools. Each tool is a standalone function that can be called and
tested independently before being wired into the agent loop.

Tools:
    search_listings(description, size, max_price)  -> list[dict]
    suggest_outfit(new_item, wardrobe)             -> str
    create_fit_card(outfit, new_item)              -> str
    estimate_price_fairness(item, listings)        -> dict   (stretch)

Failure convention: no tool raises for an expected failure. Tools that return a
string signal failure by returning a string that starts with TOOL_ERROR; the
planning loop in agent.py checks for that prefix and decides what the user sees.
"""

import os
import re
import statistics
import time

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()

# Preferred model first. Groq retired llama-4-scout from the free catalog, so a
# request for it now 404s with model_not_found; the fallbacks keep the app
# working today and let it return to scout automatically if access comes back.
# Override the whole chain with FITFINDR_MODEL in .env.
MODEL_CANDIDATES = [
    os.environ.get("FITFINDR_MODEL"),
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

# Set once the first successful call identifies a model this key can reach.
_active_model: str | None = None

# Prefix marking a string return value as a failure rather than a result.
TOOL_ERROR = "[tool_error]"


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _create_with_backoff(
    client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 3,
):
    """
    Call the model, retrying a rate limit with exponential backoff.

    Groq's free tier rate-limits bursts, which a single agent run can trigger
    on its own (search -> outfit -> fit card fires two calls back to back).
    A rate limit is temporary, so it is worth waiting out; every other error
    is raised straight away for the caller to turn into a TOOL_ERROR string.
    """
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            rate_limited = (
                "rate limit" in str(exc).lower()
                or "rate_limit" in str(exc).lower()
                or getattr(exc, "status_code", None) == 429
            )
            if not rate_limited or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, then 2s


def _call_llm(prompt: str, temperature: float, max_tokens: int = 900) -> str:
    """
    Send a single-turn prompt to Groq and return the text response.

    max_tokens is budgeted generously because the fallback models are reasoning
    models: they spend part of the allowance thinking before emitting any
    content, and a tight budget returns an empty string instead of an answer.

    Raises on failure — callers are responsible for catching and converting the
    failure into a TOOL_ERROR string, so the agent loop never sees an exception.
    """
    global _active_model

    client = _get_groq_client()
    candidates = [_active_model] if _active_model else [
        m for m in MODEL_CANDIDATES if m
    ]

    last_error: Exception | None = None
    for model in candidates:
        try:
            response = _create_with_backoff(
                client, model, prompt, temperature, max_tokens
            )
        except Exception as exc:
            # Only a missing/unavailable model is worth falling through on.
            # Auth failures would fail identically on every candidate, so
            # re-raise those immediately.
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                last_error = exc
                continue
            raise

        _active_model = model
        return (response.choices[0].message.content or "").strip()

    raise RuntimeError(
        f"No usable Groq model. Tried: {', '.join(candidates)}. "
        f"Last error: {last_error}"
    )


# ── Tool 1: search_listings ───────────────────────────────────────────────────

# Words that carry no signal about what the user wants to find.
_STOPWORDS = {
    "a", "an", "the", "for", "with", "and", "or", "of", "in", "on", "to", "my",
    "me", "i", "im", "some", "any", "that", "this", "it", "is", "am", "are",
    "looking", "look", "want", "need", "find", "get", "something", "under",
    "below", "less", "than", "over", "size", "sized", "cheap", "please",
}

# Size aliases so "M" matches a listing sized "Medium" and vice versa.
_SIZE_ALIASES = {
    "xs": {"xs", "xsmall", "extrasmall"},
    "s": {"s", "small"},
    "m": {"m", "med", "medium"},
    "l": {"l", "large"},
    "xl": {"xl", "xlarge", "extralarge"},
    "xxl": {"xxl", "xxlarge"},
}


def _tokenize(text: str) -> list[str]:
    """Lowercase a string and split it into meaningful word tokens."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    """Crude singular form so 'tees'/'jeans' match 'tee'/'jean' style tags."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _size_tokens(size: str) -> set[str]:
    """
    Split a size string into comparable tokens, expanded through the alias map.

    '"S/M"' -> {'s', 'small', 'm', 'med', 'medium'}
    '"US 8"' -> {'us', '8'}
    Token splitting matters: a plain substring test would match size 'S'
    against 'US 8'.
    """
    raw = re.findall(r"[a-z0-9]+", (size or "").lower())
    tokens: set[str] = set()
    for token in raw:
        tokens.add(token)
        for canonical, aliases in _SIZE_ALIASES.items():
            if token in aliases:
                tokens |= aliases
                tokens.add(canonical)
    return tokens


def _size_matches(wanted: str, listing_size: str) -> bool:
    """
    True if a listing's size is compatible with the size the user asked for.

    A listing marked 'One Size' matches any request — that is what one size
    means — so it is treated as a wildcard rather than filtered out.
    """
    listing_lower = (listing_size or "").lower()
    if "one size" in listing_lower:
        return True
    return bool(_size_tokens(wanted) & _size_tokens(listing_size))


def _relevance_score(listing: dict, terms: list[str]) -> int:
    """
    Score one listing against the search terms by weighted field overlap.

    Style tags and the title are the strongest signals of what a piece actually
    is; the free-text description is the weakest, since it mostly describes
    condition.
    """
    fields = (
        (" ".join(listing.get("style_tags") or []), 3),
        (listing.get("title") or "", 3),
        (listing.get("category") or "", 2),
        (" ".join(listing.get("colors") or []), 2),
        (listing.get("brand") or "", 2),
        (listing.get("description") or "", 1),
    )
    score = 0
    for text, weight in fields:
        field_terms = {_stem(t) for t in _tokenize(text)}
        for term in terms:
            if term in field_terms:
                score += weight
    return score


def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive and token-based, so "M"
                     matches "M", "S/M", "M/L" and "Medium". Listings marked
                     "One Size" match any requested size.
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform
    """
    terms = [_stem(t) for t in _tokenize(description)]
    if not terms:
        return []

    try:
        listings = load_listings()
    except (OSError, ValueError):
        # Dataset missing or corrupt — an empty result is still a valid answer,
        # and the planning loop already knows how to explain one to the user.
        return []

    scored: list[tuple[int, float, dict]] = []
    for listing in listings:
        if max_price is not None and listing.get("price", 0) > max_price:
            continue
        if size and not _size_matches(size, listing.get("size", "")):
            continue

        score = _relevance_score(listing, terms)
        if score == 0:
            continue
        # Price is the tiebreaker: same relevance, cheaper listing wins.
        scored.append((score, listing.get("price", 0.0), listing))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [listing for _, _, listing in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def _describe_item(item: dict) -> str:
    """Flatten a listing dict into a compact line for an LLM prompt."""
    parts = [
        f"Title: {item.get('title', 'Unknown item')}",
        f"Category: {item.get('category', 'unknown')}",
        f"Colors: {', '.join(item.get('colors') or []) or 'unspecified'}",
        f"Style tags: {', '.join(item.get('style_tags') or []) or 'none'}",
        f"Condition: {item.get('condition', 'unknown')}",
        f"Price: ${item.get('price', 0):.0f} on {item.get('platform', 'unknown')}",
        f"Details: {item.get('description', '')}",
    ]
    return "\n".join(parts)


def _describe_wardrobe(wardrobe: dict) -> str:
    """Format wardrobe items as a grouped list the LLM can name pieces from."""
    lines = []
    for item in wardrobe.get("items", []):
        note = f" ({item['notes']})" if item.get("notes") else ""
        lines.append(
            f"- {item.get('name', 'unnamed')} "
            f"[{item.get('category', 'unknown')}]{note}"
        )
    return "\n".join(lines)


def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handled gracefully by
                  falling back to general styling advice.

    Returns:
        A non-empty string with outfit suggestions. On an LLM failure, returns a
        string starting with TOOL_ERROR rather than raising.
    """
    if not isinstance(new_item, dict) or not new_item:
        return f"{TOOL_ERROR} No item to style — search_listings returned nothing usable."

    items = (wardrobe or {}).get("items") or []

    if items:
        prompt = (
            "You are a thrift stylist helping someone decide whether a "
            "secondhand piece is worth buying.\n\n"
            f"THE PIECE THEY FOUND:\n{_describe_item(new_item)}\n\n"
            f"WHAT THEY ALREADY OWN:\n{_describe_wardrobe(wardrobe)}\n\n"
            "Suggest 1-2 complete outfits built around the new piece. Rules:\n"
            "- Only use pieces from the list above, and name them exactly as "
            "written. Never invent clothes they do not own.\n"
            "- Each outfit needs a top, a bottom and shoes (the new piece "
            "covers one of those slots).\n"
            "- Add one concrete styling detail per outfit (how to tuck, cuff, "
            "layer or proportion it).\n"
            "- Say why the combination works in terms of color or silhouette.\n"
            "- Under 130 words, conversational, no bullet-point headers."
        )
    else:
        prompt = (
            "You are a thrift stylist. Someone found this secondhand piece but "
            "has not told you anything about their existing wardrobe.\n\n"
            f"THE PIECE:\n{_describe_item(new_item)}\n\n"
            "Give general styling advice: the kinds of pieces that pair well "
            "with it (by type and color, not brand), what silhouette to aim "
            "for, and the vibe it suits best. Open by acknowledging this is "
            "general advice because you have not seen their closet yet. "
            "Under 130 words, conversational prose only — no bullet points, "
            "no markdown headers, no bold text."
        )

    try:
        suggestion = _call_llm(prompt, temperature=0.8, max_tokens=900)
    except Exception as exc:  # network, auth, rate limit, bad response
        return (
            f"{TOOL_ERROR} Couldn't reach the styling model ({type(exc).__name__}: "
            f"{exc}). The listing details are still below — try again in a moment."
        )

    if not suggestion:
        return (
            f"{TOOL_ERROR} The styling model returned an empty response for "
            f"'{new_item.get('title', 'this item')}'. Try running the search again."
        )
    return suggestion


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption. If outfit
        is empty, whitespace-only, or itself a TOOL_ERROR string, returns a
        descriptive TOOL_ERROR string without calling the LLM.
    """
    if not isinstance(outfit, str) or not outfit.strip():
        return (
            f"{TOOL_ERROR} Can't write a fit card without an outfit suggestion — "
            "suggest_outfit returned nothing usable. Try the search again, or "
            "add a few pieces to your wardrobe first."
        )
    if outfit.strip().startswith(TOOL_ERROR):
        return (
            f"{TOOL_ERROR} Skipped the fit card because the outfit step failed "
            "upstream. Fix that first and the caption will generate."
        )
    if not isinstance(new_item, dict) or not new_item:
        return (
            f"{TOOL_ERROR} Can't write a fit card without the listing it's "
            "about — no item was passed in."
        )

    prompt = (
        "Write a caption for a thrift haul post — the kind a real person types "
        "under their outfit photo, not a product description.\n\n"
        f"THE FIND:\n{_describe_item(new_item)}\n\n"
        f"HOW THEY'RE WEARING IT:\n{outfit}\n\n"
        "Rules:\n"
        "- 2 to 4 short sentences, mostly lowercase, casual and specific.\n"
        f"- Mention the item, the ${new_item.get('price', 0):.0f} price and "
        f"{new_item.get('platform', 'the app')} once each, woven in naturally.\n"
        "- Name at least one specific piece it's styled with.\n"
        "- No hashtag walls, at most one emoji, no 'elevate your wardrobe' "
        "marketing voice.\n"
        "- Output only the caption."
    )

    try:
        # High temperature: repeated calls on the same input must read
        # differently, or the card stops feeling like something worth posting.
        card = _call_llm(prompt, temperature=1.0, max_tokens=700)
    except Exception as exc:
        return (
            f"{TOOL_ERROR} Couldn't reach the caption model ({type(exc).__name__}: "
            f"{exc}). Your outfit idea above is still good — retry for the caption."
        )

    if not card:
        return (
            f"{TOOL_ERROR} The caption model returned an empty response. "
            "Try again — the outfit suggestion above is unaffected."
        )
    return card


# ── Tool 4: estimate_price_fairness (stretch) ─────────────────────────────────

def estimate_price_fairness(
    item: dict,
    listings: list[dict] | None = None,
) -> dict:
    """
    Estimate whether a listing's price is fair against comparable listings.

    A comparable is another listing in the same category that shares at least
    one style tag. Deterministic — no LLM call — so the verdict is testable.

    Args:
        item:     The listing dict to evaluate.
        listings: Pool of listings to compare against. None loads the full
                  dataset via load_listings().

    Returns:
        A dict with keys:
            verdict            (str): "great deal", "fair", "a bit high",
                                      or "not enough data"
            item_price       (float): the item's asking price
            median_comparable (float | None): median price of comparables
            comparable_count   (int): how many comparables were found
            message            (str): one-sentence human-readable summary

    Never raises: with fewer than 3 comparables it returns the
    "not enough data" verdict rather than failing.
    """
    price = float((item or {}).get("price", 0.0) or 0.0)
    no_data = {
        "verdict": "not enough data",
        "item_price": price,
        "median_comparable": None,
        "comparable_count": 0,
        "message": (
            "Not enough comparable listings in the dataset to judge this price "
            "fairly — treat the asking price as unverified."
        ),
    }

    if not isinstance(item, dict) or not item:
        return no_data

    if listings is None:
        try:
            listings = load_listings()
        except (OSError, ValueError):
            return no_data

    category = item.get("category")
    tags = set(item.get("style_tags") or [])

    comparables = [
        other.get("price")
        for other in listings
        if other.get("id") != item.get("id")
        and other.get("category") == category
        and (tags & set(other.get("style_tags") or []))
        and isinstance(other.get("price"), (int, float))
    ]

    if len(comparables) < 3:
        no_data["comparable_count"] = len(comparables)
        return no_data

    median = round(statistics.median(comparables), 2)
    ratio = price / median if median else 1.0

    if ratio <= 0.8:
        verdict = "great deal"
        gloss = f"about {round((1 - ratio) * 100)}% below"
    elif ratio <= 1.2:
        verdict = "fair"
        gloss = "right around"
    else:
        verdict = "a bit high"
        gloss = f"about {round((ratio - 1) * 100)}% above"

    return {
        "verdict": verdict,
        "item_price": price,
        "median_comparable": median,
        "comparable_count": len(comparables),
        "message": (
            f"${price:.2f} is {gloss} the ${median:.2f} median for "
            f"{len(comparables)} comparable {category} listings — {verdict}."
        ),
    }
