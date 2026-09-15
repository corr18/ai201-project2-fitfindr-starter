# FitFindr — planning.md

> Spec written before implementation. Updated once before starting stretch features
> (Tool 4: `estimate_price_fairness`, and retry-with-loosened-constraints in the planning loop) —
> those two sections are marked **[added in stretch update]**.

---

## What FitFindr Does (in my own words)

FitFindr takes one natural-language thrifting request, pulls the description, size, and price
ceiling out of it, and searches a local listings dataset for matches. If it finds something, it
takes the single best listing and asks an LLM how to wear it with the pieces the user already
owns, then turns that styling advice into a caption the user could actually post. Each tool is
triggered by the output of the one before it: search runs on the parsed query, outfit suggestion
runs only when search returned at least one listing, and the fit card runs only when the outfit
suggestion produced real text. When a step fails, the agent stops at that step, writes a specific
message explaining what it tried and what the user could change, and returns without calling the
downstream tools on empty input.

---

## Tools

### Tool 1: search_listings

**What it does:**
Filters the 40-listing mock dataset by price and size, then scores the survivors on keyword
overlap between the user's description and each listing's title, description, style tags, colors,
brand, and category. Returns the surviving listings ranked best-match-first.

**Input parameters:**
- `description` (str): free-text keywords describing the wanted item, e.g. `"vintage graphic tee"`.
  Tokenized to lowercase words; stopwords (`a`, `the`, `for`, `under`, `size`, …) are dropped.
- `size` (str | None): size to filter by, e.g. `"M"`. Matching is case-insensitive and
  **substring-aware in both directions**, because dataset sizes are messy: `"M"` must match
  `"M"`, `"S/M"`, `"M/L"`, and `"Medium"`. `None` skips size filtering entirely.
- `max_price` (float | None): inclusive price ceiling in dollars. `None` skips price filtering.

**What it returns:**
`list[dict]` — the raw listing dicts from `data/listings.json`, sorted by descending relevance
score. Each dict carries `id` (str), `title` (str), `description` (str), `category` (str),
`style_tags` (list[str]), `size` (str), `condition` (str), `price` (float), `colors` (list[str]),
`brand` (str | None), `platform` (str). Listings scoring 0 (no keyword overlap at all) are dropped,
so every returned listing is relevant, not merely price/size-legal. Returns `[]` when nothing
matches — never raises, never returns `None`.

**What happens if it fails or returns nothing:**
Returns an empty list. The planning loop, not the tool, decides what to do next: it retries with
loosened constraints (see the stretch retry section below), and if that still returns nothing it
sets `session["error"]` to a message naming the three filters it used and suggesting a concrete
relaxation, then returns before `suggest_outfit` is ever called.

---

### Tool 2: suggest_outfit

**What it does:**
Asks Groq's `meta-llama/llama-4-scout-17b-16e-instruct` how to wear one specific listing, either
with the user's own named wardrobe pieces or — when the wardrobe is empty — as general styling
advice about what to pair the item with.

**Input parameters:**
- `new_item` (dict): one listing dict from `search_listings`. The prompt uses its `title`,
  `description`, `category`, `style_tags`, `colors`, `condition`, `price`, and `platform`.
- `wardrobe` (dict): a dict with an `items` key holding a list of wardrobe item dicts, each with
  `id`, `name`, `category`, `colors`, `style_tags`, and optional `notes`. May be `{"items": []}`.

**What it returns:**
`str` — a non-empty styling suggestion. With a populated wardrobe: 1–2 outfit combinations that
name actual wardrobe pieces ("your baggy straight-leg jeans and chunky white sneakers") plus a
concrete styling detail. With an empty wardrobe: general advice describing the *kinds* of pieces
that pair with the item and the vibe it suits, explicitly framed as generic because no wardrobe
was provided.

**What happens if it fails or returns nothing:**
Two separate failures. (1) **Empty wardrobe** is not an error — it switches to the general-advice
prompt and the agent tells the user in the UI that advice is generic. (2) **LLM call fails**
(no API key, network error, rate limit) or returns an empty string — the tool catches the
exception and returns a string that starts with the sentinel prefix `[tool_error]`, naming what
broke. The planning loop checks for that prefix, stores the message in `session["error"]`, and
stops before `create_fit_card`.

---

### Tool 3: create_fit_card

**What it does:**
Turns the outfit suggestion plus the listing into a 2–4 sentence caption that reads like a real
OOTD post — the thing a person would actually type under the photo.

**Input parameters:**
- `outfit` (str): the string returned by `suggest_outfit`. Must be non-empty and
  non-whitespace, and must not be a `[tool_error]` string.
- `new_item` (dict): the same listing dict, so the caption can name the item, its price, and the
  platform once each.

**What it returns:**
`str` — a 2–4 sentence lowercase-leaning caption. Generated at `temperature=1.0` so repeated calls
on identical input produce visibly different captions; different items produce different captions
because item title, price, platform, and outfit text are all in the prompt.

**What happens if it fails or returns nothing:**
If `outfit` is empty, whitespace-only, or a `[tool_error]` string, the tool returns
`"[tool_error] Can't write a fit card without an outfit suggestion — ..."` immediately and makes
no LLM call. If the LLM call itself fails, the same `[tool_error]` prefix is returned. Either way
it returns a descriptive string and never raises. The planning loop surfaces it as
`session["error"]` and leaves `session["fit_card"]` as `None`.

---

### Additional Tools

### Tool 4: estimate_price_fairness  **[added in stretch update]**

**What it does:**
Judges whether a listing's asking price is fair by comparing it against comparable listings in the
same dataset — same category, overlapping style tags — with no LLM call, so it is deterministic
and testable.

**Input parameters:**
- `item` (dict): the listing dict to evaluate. Uses its `price`, `category`, and `style_tags`.
- `listings` (list[dict] | None): the pool to compare against. `None` means load the full dataset
  with `load_listings()`. Passing a pool explicitly is what makes the tool unit-testable.

**What it returns:**
`dict` with keys: `verdict` (str — one of `"great deal"`, `"fair"`, `"a bit high"`,
`"not enough data"`), `item_price` (float), `median_comparable` (float | None),
`comparable_count` (int), `message` (str — one human-readable sentence naming the median and the
number of comparables). The verdict comes from the ratio of item price to the median comparable:
≤0.8 is a great deal, ≤1.2 is fair, above that is a bit high.

**What happens if it fails or returns nothing:**
Fewer than 3 comparables is a real possibility in a 40-item dataset, so it is handled as a first-
class outcome rather than an error: `verdict` becomes `"not enough data"`, `median_comparable`
becomes `None`, and `message` says so plainly. The planning loop treats this tool as advisory —
it never terminates the run, because a missing price opinion should not cost the user their
outfit and fit card.

---

## Planning Loop

The loop runs as a sequence of guarded steps over one session dict. Each step decides whether to
run, and every step can end the run early. It is not a fixed 3-call pipeline: the number of tool
calls in a run varies from 1 (search found nothing even after retry) to 4 (search, price check,
outfit, fit card), depending entirely on what came back.

**Step 1 — Parse.** Regex over the raw query extracts three things:
`under $30` / `below 30` / `$30` → `max_price = 30.0`; `size M` / `size 8` / a bare
`M`/`S`/`L`/`XL` token → `size`; everything left after stripping the price and size phrases and a
few lead-ins ("looking for", "i want") → `description`. If `description` comes out empty, the
loop sets `session["error"]` to a prompt for more detail and returns after **zero** tool calls.

**Step 2 — Search.** Call `search_listings(description, size, max_price)`. Branch on the result:
- **Non-empty** → store in `session["search_results"]`, continue to step 3.
- **Empty and a size filter was applied** → **[stretch: retry]** retry as
  `search_listings(description, None, max_price)`, record
  `session["retry"] = {"dropped": "size", ...}`. If that returns results, continue to step 3 with
  a note that the size filter was dropped.
- **Empty and a price filter was applied** → **[stretch: retry]** retry once more with both size
  and price dropped, record `session["retry"]`, continue if it now returns results.
- **Still empty** → set `session["error"]` naming the description, size, and price actually used
  and suggesting a specific relaxation, then `return session`. `suggest_outfit` is never reached.

**Step 3 — Select.** `session["selected_item"] = session["search_results"][0]` (highest relevance
score). This dict is the single object handed to every downstream tool — nothing is re-derived
from the query string after this point.

**Step 4 — Price check.** **[added in stretch update]** Call
`estimate_price_fairness(selected_item)` and store the dict in `session["price_check"]`. This step
never terminates the run: a `"not enough data"` verdict is stored and the loop continues.

**Step 5 — Suggest outfit.** Call `suggest_outfit(selected_item, session["wardrobe"])`. If the
returned string starts with `[tool_error]`, copy it into `session["error"]` and `return session`
with `fit_card` still `None`. Otherwise store it in `session["outfit_suggestion"]`. If the
wardrobe was empty, also set `session["notes"]` to flag that the advice is generic.

**Step 6 — Fit card.** Call `create_fit_card(session["outfit_suggestion"], selected_item)`. On a
`[tool_error]` string, set `session["error"]` and leave `session["fit_card"]` as `None` — the
outfit suggestion from step 5 is still returned and still shown, because a failed caption should
not discard work that succeeded. Otherwise store the caption.

**Step 7 — Return** the session dict. The caller checks `session["error"]` first.

**How it knows it's done:** the loop is done when `fit_card` is set, or when any guarded step has
written to `session["error"]` and returned early. There is no open-ended re-planning — the
branching is over tool *results*, not over LLM-chosen next actions.

---

## State Management

One mutable `session` dict, created by `_new_session(query, wardrobe)`, is the only state. It is
created at the top of `run_agent`, threaded through every step, and returned to the caller. Nothing
is stored in module-level globals, so two concurrent users cannot interfere.

| Key | Written by | Read by |
|-----|-----------|---------|
| `query` | caller | step 1 (parse) |
| `parsed` | step 1 | step 2, error messages |
| `search_results` | step 2 | step 3 |
| `retry` | step 2 (stretch) | UI note, error messages |
| `selected_item` | step 3 | steps 4, 5, 6 and the UI |
| `wardrobe` | caller | step 5 |
| `price_check` | step 4 (stretch) | UI |
| `outfit_suggestion` | step 5 | step 6 and the UI |
| `fit_card` | step 6 | the UI |
| `notes` | steps 2, 5 | the UI |
| `error` | any step | checked first by every caller |

The key state-passing claim: `suggest_outfit` receives the *exact same dict object* that
`search_listings` returned and that `session["selected_item"]` points at — verified in
`agent.py`'s CLI block with an `is` identity check, not just an equality check. The user never
re-enters the item, and no step re-parses the original query string after step 1.

---

## Error Handling

| Tool | Failure mode | Agent response |
|------|-------------|----------------|
| (parse) | Query has no describable item, e.g. `"under $20"` | "I couldn't tell what you're looking for from that. Try naming the piece — e.g. 'vintage graphic tee under $30, size M'." Returns after zero tool calls. |
| search_listings | No results match the query | Retries automatically with the size filter dropped, then with the price filter dropped, and says which one it dropped. If still nothing: "No listings matched 'designer ballgown' with size XXS under $5. I also tried without the size filter and without the price cap. The dataset tops out at $75 and has no formalwear — try a different piece, or raise your price cap." Does not call `suggest_outfit`. |
| suggest_outfit | Wardrobe is empty | Not an error — switches to a general-styling prompt and returns advice about what kinds of pieces pair with the item, and the UI notes "based on general styling, since no wardrobe was provided." |
| suggest_outfit | LLM call fails / returns empty | Returns `"[tool_error] Couldn't reach the styling model (<reason>). The listing is still below — try again in a moment."` The loop stops before `create_fit_card` and shows the found listing anyway. |
| create_fit_card | Outfit input is missing or incomplete | Returns `"[tool_error] Can't write a fit card without an outfit suggestion — suggest_outfit returned nothing usable."` without making an LLM call. The outfit panel keeps whatever step 5 produced; only the fit-card panel shows the error. |
| create_fit_card | LLM call fails | Same `[tool_error]` prefix naming the reason; `session["fit_card"]` stays `None` and the listing + outfit are still returned. |
| estimate_price_fairness | Fewer than 3 comparable listings | Returns `verdict="not enough data"` with a message saying so. Advisory only — the run continues to the outfit and fit card. |

---

## Architecture

```
                          ┌──────────────────────────────┐
  User query ───────────► │  run_agent(query, wardrobe)  │
  + wardrobe choice       └──────────────┬───────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  session = _new_session(...)  │◄───────────┐
                          └──────────────┬───────────────┘            │
                                         │                   all steps read/write
   PLANNING LOOP                         │                   this one dict
       │                                 │                             │
       │  Step 1: parse_query(query)     │                             │
       ├────────────────────────────────►│                             │
       │      description == ""          │                             │
       │      └──► [ERROR] "name the piece" ──────────────────────────►│──► return
       │                                 │                             │
       │      Session: parsed = {description, size, max_price}         │
       │                                 │                             │
       │  Step 2: search_listings(description, size, max_price)        │
       ├────────────────────────────────►│                             │
       │      results == []  &&  size set │                            │
       │      └─► RETRY search_listings(description, None, max_price)  │
       │             results == [] && max_price set                    │
       │             └─► RETRY search_listings(description, None, None)│
       │                    Session: retry = {"dropped": ...}          │
       │      results == [] (after retries)                            │
       │      └──► [ERROR] "no listings matched ..." ─────────────────►│──► return
       │                                 │                             │
       │      results == [item, ...]     │                             │
       │      Session: search_results = results                        │
       │                                 │                             │
       │  Step 3: select                 │                             │
       │      Session: selected_item = search_results[0]  ────────────►│
       │                                 │                             │
       │  Step 4: estimate_price_fairness(selected_item)   [advisory]  │
       ├────────────────────────────────►│                             │
       │      Session: price_check = {verdict, median_comparable, ...} │
       │      (verdict "not enough data" does NOT stop the run)        │
       │                                 │                             │
       │  Step 5: suggest_outfit(selected_item, wardrobe)              │
       ├────────────────────────────────►│                             │
       │      returns "[tool_error] ..."  │                            │
       │      └──► [ERROR] + keep listing ────────────────────────────►│──► return
       │                                 │                             │
       │      Session: outfit_suggestion = "..."  ────────────────────►│
       │                                 │                             │
       │  Step 6: create_fit_card(outfit_suggestion, selected_item)    │
       ├────────────────────────────────►│                             │
       │      returns "[tool_error] ..."  │                            │
       │      └──► [ERROR] + keep listing & outfit ───────────────────►│──► return
       │                                 │                             │
       │      Session: fit_card = "..."  ─────────────────────────────►│
       │                                 │                             │
       │  Step 7: return session ────────┴─────────────────────────────┘
       ▼
  app.py handle_query() maps session → three Gradio panels
  (checks session["error"] FIRST)
```

---

## AI Tool Plan

**Tool used throughout: Claude (Claude Code in VS Code)**, because it can read the actual repo
files and run the tests it writes, so I can check generated code against the dataset instead of
against my memory of the dataset.

**Milestone 3 — Individual tool implementations:**

- *search_listings:* Give Claude the **Tool 1** block above (all three parameters with types, the
  return-value description, and the empty-list failure mode) plus `utils/data_loader.py`, and ask
  it to implement the function using `load_listings()` rather than re-opening the JSON. **Verify
  before trusting:** confirm it filters on all three parameters, that size matching is
  bidirectional substring (I will specifically check `size="M"` matches the `"S/M"` listing,
  because a naive `==` will silently drop half the dataset), that zero-score listings are dropped,
  and that it returns `[]` rather than raising. Then run it against three queries — one that
  should hit, one price-impossible, one size-only.
- *suggest_outfit:* Give Claude the **Tool 2** block and the wardrobe schema from
  `data/wardrobe_schema.json`, and ask for two prompt paths (populated vs. empty wardrobe) with the
  Groq call wrapped in `try/except`. **Verify:** call it with `get_empty_wardrobe()` and confirm no
  exception and a non-empty string; call it with `get_example_wardrobe()` and confirm the output
  names actual wardrobe pieces rather than inventing clothes the user doesn't own.
- *create_fit_card:* Give Claude the **Tool 3** block including the "sounds like a caption, not a
  product description" requirement and the empty-outfit guard. **Verify:** run it three times on
  identical input and diff the outputs — if they match, raise temperature; pass `""` as the outfit
  and confirm a `[tool_error]` string comes back with no LLM call made.

**Milestone 4 — Planning loop and state management:**

Give Claude the **Architecture** diagram above verbatim, plus the **Planning Loop** and
**State Management** sections, and ask it to implement `run_agent()` in `agent.py` against the
existing `_new_session()` shape. **Verify before running:** every early-return branch in the
diagram exists in the code; `search_listings` results are branched on rather than assumed;
nothing is written to a module-level global; `suggest_outfit` is genuinely unreachable when
`search_results` is empty. Then prove state passing with an `is` identity assertion that
`session["selected_item"]` is the same object handed to `suggest_outfit`, and run the
impossible query to confirm `error` is set and `fit_card` is still `None`.

**What I expect to override:** generated search code tends to use exact size equality and to
`raise` on missing API keys mid-loop. Both get rewritten — sizes are substring-matched, and every
LLM failure returns a `[tool_error]` string so the loop, not an exception, decides what the user
sees.

---

## A Complete Interaction (Step by Step)

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans
and chunky sneakers. What's out there and how would I style it?"

**Step 1 — Parse.** `parse_query()` finds `under $30` → `max_price = 30.0`; finds no size token →
`size = None`; strips the price phrase and the "i'm looking for" lead-in → `description = "vintage
graphic tee"`. Session now holds
`parsed = {"description": "vintage graphic tee", "size": None, "max_price": 30.0}`.
Description is non-empty, so the loop continues.

**Step 2 — Search.** `search_listings("vintage graphic tee", None, 30.0)`. Price filter keeps
listings ≤ $30; no size filter; keyword scoring over `vintage`, `graphic`, `tee` ranks the band tee
and the Y2K butterfly baby tee at the top (both carry the `graphic tee` and `vintage` style tags).
Returns a non-empty list, so no retry is needed. Session: `search_results = [<band tee>, <baby
tee>, ...]`.

**Step 3 — Select.** `selected_item = search_results[0]` — the top-scoring tee, e.g. *"Vintage Band
Tee — Faded Black, $24, depop, good condition."* This exact dict is what every later step uses.

**Step 4 — Price check (stretch).** `estimate_price_fairness(<band tee>)` compares $24 against the
median price of other `tops` sharing its style tags. Session:
`price_check = {"verdict": "fair", "median_comparable": 26.0, "comparable_count": 9, ...}`.

**Step 5 — Suggest outfit.** `suggest_outfit(<band tee>, <example wardrobe>)` sends the item plus
the 10 named wardrobe pieces to the LLM. Returns something like: *"Wear it with your baggy
straight-leg jeans and chunky white sneakers, then throw the vintage black denim jacket over the
top — the faded black tee and black jacket keep it tonal while the light wash denim breaks it up.
Half-tuck the front hem so the high waist actually reads."* Session: `outfit_suggestion = "..."`.
No `[tool_error]` prefix, so the loop continues.

**Step 6 — Fit card.** `create_fit_card(<that outfit string>, <band tee>)` — note it receives the
step 5 output and the step 3 item, neither re-entered by the user. Returns e.g.: *"found this
faded band tee on depop for $24 and it's already the most worn thing i own. baggy jeans, chunky
sneakers, denim jacket over the top. no notes."* Session: `fit_card = "..."`.

**Step 7 — Return.** `error` is `None`, so the caller renders all three panels.

**Final output to user:** Three panels in the Gradio UI — the listing (title, price, platform,
size, condition, plus the price-fairness verdict), the outfit idea naming their own wardrobe
pieces, and the shareable caption. Four tool calls total for this query; an impossible query like
*"designer ballgown size XXS under $5"* makes three (search plus two loosened retries), fills only
the first panel with an error explaining what was tried, and never touches the LLM.
