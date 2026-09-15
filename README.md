# FitFindr 🛍️

A multi-tool AI agent that finds secondhand clothing listings, works out how to wear them with
what you already own, and writes the caption you'd post it with.

You type one sentence — *"vintage graphic tee under $30, size M"* — and the agent parses it,
searches a 40-listing dataset, checks whether the asking price is fair, styles the item against
your wardrobe, and writes a shareable fit card. When a step fails it says what broke and what to
try instead, and it keeps whatever already succeeded.

> **Demo video:** ⚠️ *TODO — record a 3–5 minute walkthrough and paste the link here before
> submitting.* See [What to show in the demo](#what-to-show-in-the-demo) for the shot list.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# source .venv/Scripts/activate  # Windows (Git Bash)

pip install -r requirements.txt
```

Create a `.env` file in the repo root (already gitignored):

```
GROQ_API_KEY=your_key_here
```

Run it:

```bash
python app.py            # Gradio UI — open the URL printed in your terminal
python agent.py          # CLI walkthrough of all five paths
python -m pytest tests/  # 48 tests
```

### A note on the model

The assignment specifies `meta-llama/llama-4-scout-17b-16e-instruct`. Groq has since retired that
model from the free catalog — requesting it now returns
`404 model_not_found`. Rather than hardcode a substitute, `tools.py` keeps scout as the **first**
entry in `MODEL_CANDIDATES` and falls through to the next available model only on a
`model_not_found` error, caching whichever one works. Rate limits are retried with backoff first,
and auth errors are raised immediately rather than triggering the fallback, since a bad key would
fail identically on every candidate. If scout access returns, the agent uses it again with no code change. To pin a specific
model, set `FITFINDR_MODEL=...` in `.env`. Runs during development resolved to
`openai/gpt-oss-120b`.

---

## Tool Inventory

All four tools live in [tools.py](tools.py). Signatures below are copied from the code.

### 1. `search_listings(description, size, max_price) -> list[dict]`

Searches the mock listings dataset and returns matches ranked best-first.

| Parameter | Type | Meaning |
|---|---|---|
| `description` | `str` | Free-text keywords, e.g. `"vintage graphic tee"`. Tokenized to lowercase words with stopwords removed. |
| `size` | `str \| None` | Size filter, e.g. `"M"`. `None` skips size filtering. Default `None`. |
| `max_price` | `float \| None` | Inclusive price ceiling. `None` skips price filtering. Default `None`. |

**Returns:** `list[dict]` — raw listing dicts sorted by descending relevance score, price
ascending as the tiebreaker. Each dict has `id` (str), `title` (str), `description` (str),
`category` (str), `style_tags` (list[str]), `size` (str), `condition` (str), `price` (float),
`colors` (list[str]), `brand` (str | None), `platform` (str). Listings scoring zero are dropped,
so every result is actually relevant rather than merely price- and size-legal. Returns `[]` when
nothing matches.

**Purpose:** the entry point of every interaction — it's what turns a sentence into candidate items.

Two details that matter more than they look:

- **Size matching is token-based, not substring.** Dataset sizes are messy (`"S/M"`, `"M/L"`,
  `"US 8.5"`, `"XL (oversized)"`). A substring test would match requested size `"S"` against
  `"US 8"`. Sizes are split into tokens and expanded through an alias map, so `"M"` matches `"M"`,
  `"S/M"`, `"M/L"` and `"Medium"` but not `"XL (oversized)"`.
- **`"One Size"` listings match any requested size**, because that is what one size means.

### 2. `suggest_outfit(new_item, wardrobe) -> str`

Asks the LLM how to wear one listing, using the user's actual wardrobe when there is one.

| Parameter | Type | Meaning |
|---|---|---|
| `new_item` | `dict` | A listing dict from `search_listings`. Uses its title, description, category, style tags, colors, condition, price, platform. |
| `wardrobe` | `dict` | Dict with an `items` key holding wardrobe item dicts (`id`, `name`, `category`, `colors`, `style_tags`, optional `notes`). May be `{"items": []}`. |

**Returns:** `str` — a non-empty styling suggestion. With a populated wardrobe: 1–2 outfits naming
pieces the user actually owns, each with a concrete styling detail and a reason it works. With an
empty wardrobe: general advice about what *kinds* of pieces pair with the item, opening with an
acknowledgement that it hasn't seen their closet. On an LLM failure it returns a string prefixed
`[tool_error]` — it never raises.

**Purpose:** the reasoning step — turns "here is an item" into "here is how it fits your life."

### 3. `create_fit_card(outfit, new_item) -> str`

Writes a short shareable caption for the find.

| Parameter | Type | Meaning |
|---|---|---|
| `outfit` | `str` | The suggestion string from `suggest_outfit`. |
| `new_item` | `dict` | The same listing dict, so the caption can name the item, price and platform. |

**Returns:** `str` — a 2–4 sentence, mostly-lowercase caption. Generated at `temperature=1.0`, so
two calls on identical input produce different captions (asserted in
`test_fit_card_varies_between_calls`). Returns a `[tool_error]` string if `outfit` is empty,
whitespace-only, or itself a `[tool_error]` string — in that case no LLM call is made at all.

**Purpose:** the payoff — the thing the user would actually post.

### 4. `estimate_price_fairness(item, listings) -> dict` *(stretch)*

Judges whether an asking price is fair against comparable listings. No LLM call, so it's
deterministic and unit-testable.

| Parameter | Type | Meaning |
|---|---|---|
| `item` | `dict` | The listing to evaluate. Uses `price`, `category`, `style_tags`. |
| `listings` | `list[dict] \| None` | Comparison pool. `None` loads the full dataset via `load_listings()`. Default `None`. |

**Returns:** `dict` with `verdict` (str: `"great deal"` / `"fair"` / `"a bit high"` /
`"not enough data"`), `item_price` (float), `median_comparable` (float | None),
`comparable_count` (int), `message` (str). A comparable is another listing in the same category
sharing at least one style tag; the verdict comes from the ratio of price to median (≤0.8 great
deal, ≤1.2 fair, above that a bit high).

**Purpose:** answers the question that actually decides a thrift purchase — *is this a good price?*

### Supporting functions in [agent.py](agent.py)

- `parse_query(query: str) -> dict` — returns `{"description": str, "size": str | None,
  "max_price": float | None}`. Regex, not an LLM: the fields are shallow and regular, so keeping
  it deterministic means a parse never costs an API call and never fails differently on two
  identical runs.
- `run_agent(query: str, wardrobe: dict) -> dict` — the planning loop; returns the session dict.

---

## How the Planning Loop Works

`run_agent()` in [agent.py](agent.py#L141) is a sequence of guarded steps over one session dict.
Each step decides whether to run based on what the previous one returned, and any step can end the
run. **A run makes between zero and four tool calls depending on the input** — that variation is
the loop working, and it's asserted in
`test_agent_takes_different_paths_for_different_queries`.

**Step 1 — Parse.** `parse_query()` pulls description, size and max price out of the raw text.
*Branch:* if the description is empty, or contains nothing but filler words once stopwords are
removed (`"I want something under $20"` → `"something"` → nothing), set `error` and **return
having called zero tools.**

**Step 2 — Search.** Call `search_listings(description, size, max_price)`. *Branch on the result:*
- Non-empty → store and continue.
- Empty **and** a size was given → **retry** without the size filter (stretch feature). Size is the
  constraint most likely to be the sole reason a genuine match was excluded.
- Still empty **and** a price cap was given → **retry** with both size and price dropped.
- Still empty after both retries → set `error` naming the exact constraints used, and **return
  before `suggest_outfit` is ever reached.**

Each successful retry records what was dropped in `session["retry"]` and appends a plain-English
note to `session["notes"]`, which the UI shows — the agent never silently changes what you asked for.

**Step 3 — Select.** `selected_item = search_results[0]`. This exact dict object is what every
downstream tool receives.

**Step 4 — Price check** *(stretch)*. `estimate_price_fairness(selected_item)`. **Advisory only —
it can never end the run.** A `"not enough data"` verdict is stored and the loop continues, because
a missing price opinion shouldn't cost the user their outfit and fit card.

**Step 5 — Suggest outfit.** *Branch:* if the returned string starts with `[tool_error]`, copy it
into `error` and return — but **keep the listing**, so the user still sees what was found. Otherwise
store it, and if the wardrobe was empty add a note that the advice is generic.

**Step 6 — Fit card.** *Branch:* on a `[tool_error]` string, set `error` and leave `fit_card` as
`None` — **without discarding the outfit suggestion from step 5.** A failed caption shouldn't throw
away work that succeeded.

**Step 7 — Return** the session.

The loop is done when `fit_card` is set, or when a guarded step wrote to `error` and returned early.
`session["tool_calls"]` logs the path actually taken, so you can read off any run's decisions:

```
happy path:  ['search_listings -> 20 results', 'estimate_price_fairness -> a bit high',
              'suggest_outfit -> ok', 'create_fit_card -> ok']

no results:  ['search_listings -> 0 results',
              'search_listings (retry, no size) -> 0 results',
              'search_listings (retry, no size/price) -> 0 results']

unparseable: []
```

---

## State Management

One mutable `session` dict, created by `_new_session(query, wardrobe)`, is the only state. It's
created at the top of `run_agent`, threaded through every step, and returned to the caller. Nothing
lives in module-level globals, so concurrent users can't interfere.

| Key | Written by | Read by |
|---|---|---|
| `query` | caller | step 1 |
| `parsed` | step 1 | step 2, error messages |
| `search_results` | step 2 | step 3, UI result count |
| `retry` | step 2 | UI notes |
| `selected_item` | step 3 | steps 4, 5, 6, UI |
| `wardrobe` | caller | step 5 |
| `price_check` | step 4 | UI |
| `outfit_suggestion` | step 5 | step 6, UI |
| `fit_card` | step 6 | UI |
| `notes` | steps 2, 5 | UI |
| `tool_calls` | every step | debugging, demo |
| `error` | any step | checked first by every caller |

**The claim that matters:** `suggest_outfit` receives *the same dict object* that `search_listings`
returned — not a copy, and not something re-derived from the query string. That's verified by
identity, not equality, both in `agent.py`'s CLI block and in
`test_state_passes_the_same_object_between_tools`:

```python
assert seen["outfit_item"] is session["selected_item"]
assert seen["card_item"]   is session["selected_item"]
assert session["selected_item"] is session["search_results"][0]
```

The user never re-enters the item, and after step 1 nothing re-reads the original query string.

---

## Error Handling

Every tool owns its failure mode and returns a value describing it. **No tool raises for an expected
failure**, so the planning loop — not an exception handler — decides what the user sees. String-
returning tools signal failure with a `[tool_error]` prefix; the loop strips that prefix before
showing the message.

| Tool | Failure mode | What the agent does |
|---|---|---|
| *(parse)* | Nothing describable in the query | *"I couldn't tell what you're looking for from that. Try naming the piece itself — for example 'vintage graphic tee under $30, size M'."* Zero tool calls. |
| `search_listings` | No results | Retries without size, then without price, telling the user which filter it dropped. If still empty: an error naming the exact constraints used plus the dataset's real bounds. **Does not call `suggest_outfit`.** |
| `search_listings` | Dataset file missing or corrupt | Returns `[]` — caught as `OSError`/`ValueError`. Degrades into the no-results path rather than crashing the app. |
| `suggest_outfit` | Wardrobe is empty | **Not an error.** Switches to a general-styling prompt; the UI notes the advice is generic. |
| `suggest_outfit` | LLM unreachable / empty response | `[tool_error]` naming the exception type. Loop stops before the fit card but **still shows the listing**. |
| *(both LLM tools)* | Rate limited (HTTP 429) | Retried up to 3 times with exponential backoff (1s, 2s) inside `_create_with_backoff()` before giving up — a rate limit is temporary, so it's worth waiting out rather than surfacing as a failure. |
| `create_fit_card` | Outfit empty, whitespace, or an upstream `[tool_error]` | Returns a descriptive `[tool_error]` string **without calling the LLM** — no point spending a request on input that can't work. |
| `create_fit_card` | LLM unreachable | `[tool_error]`; `fit_card` stays `None` but the **listing and outfit are still returned**. |
| `estimate_price_fairness` | Fewer than 3 comparables | `verdict="not enough data"`, `median_comparable=None`. Advisory — the run continues. |

### Concrete example from testing

Running the deliberate no-results query:

```bash
$ python -c "from tools import search_listings; print(search_listings('designer ballgown', size='XXS', max_price=5))"
[]
```

Empty list, no exception. Through the full agent:

```
$ python agent.py
=== No-results path ===

Error message: No listings matched "designer ballgown", size XXS, under $5. I also retried
without the size and price filters and still found nothing, so it's the description that's the
problem. Try different words for the piece (the dataset is tops, bottoms, outerwear, shoes and
accessories, $12–$75), or search for something adjacent — "denim jacket", "graphic tee",
"wide-leg trousers".

Tools called: ['search_listings -> 0 results', 'search_listings (retry, no size) -> 0 results',
               'search_listings (retry, no size/price) -> 0 results']
fit_card is None: True
outfit_suggestion is None: True
```

The message names all three constraints it used, states that it already retried without them (so
the user doesn't repeat that experiment), gives the dataset's actual categories and price range,
and suggests three concrete alternatives. `suggest_outfit` was never called.

### A failure I hit for real

While testing, `suggest_outfit` returned a `[tool_error]` mid-run after several back-to-back agent
invocations hit Groq's free-tier rate limit:

```
Tools called: ['search_listings -> 1 results', 'estimate_price_fairness -> not enough data',
               'suggest_outfit -> failed']
```

The agent didn't crash, didn't call `create_fit_card` on an error string, and kept the found
listing — the same handling that covers a missing API key covered a failure I hadn't planned for.

But degrading gracefully was the wrong final answer here. A rate limit is *temporary*, unlike a
missing key, and one agent run fires two LLM calls back to back, so it can trip the limit by
itself. I added `_create_with_backoff()`: rate limits get up to 3 attempts with exponential backoff
(1s, then 2s), while every other error still raises immediately for the caller to turn into a
`[tool_error]`. Before that change the full test suite failed intermittently on whichever LLM test
ran fourth; it now passes consistently. Good error handling meant distinguishing *retry this* from
*report this*, not treating every failure the same way.

---

## Testing

48 tests across [tests/test_tools.py](tests/test_tools.py) (tool behavior and failure modes) and
[tests/test_agent.py](tests/test_agent.py) (loop branching and state passing).

```bash
python -m pytest tests/              # all 48
python -m pytest tests/ -m "not llm" # 44 deterministic tests, no API key needed
```

Use `python -m pytest`, not bare `pytest` — the `-m` form puts the repo root on `sys.path`, which
is what makes `from tools import search_listings` resolve.

The four tests marked `llm` make real API calls and skip automatically when `GROQ_API_KEY` is
unset. The loop tests stub `suggest_outfit` and `create_fit_card`, so branching is tested
deterministically and offline — the loop's decisions are the thing under test, not the model's
prose.

---

## Spec Reflection

**Where the spec helped.** Writing the error-handling table in `planning.md` before any code turned
error handling into a design decision instead of an afterthought. Deciding in advance that a failed
`create_fit_card` should *keep* the outfit suggestion — rather than returning one blanket error —
is what produced the partial-success path in `handle_query()`. If I'd written the loop first I'd
almost certainly have done the easy thing: one `error` field, one failed run, three empty panels.
The `[tool_error]` prefix convention also came directly from the spec's insistence that every tool
returns rather than raises, and it's what lets the loop branch on failures uniformly instead of
wrapping each call in its own `try`.

**Where implementation diverged.**

1. **Size matching got much more complicated than specced.** The spec said "case-insensitive
   matching, e.g. `M` matches `S/M`" and I assumed a substring check. It's wrong: requested size
   `"S"` is a substring of `"US 8"`, so asking for a small shirt returns size-8 shoes. The
   implementation tokenizes both sides and expands an alias map instead, and `"One Size"` became an
   explicit wildcard — a case the spec didn't consider at all. There's a parametrized test pinning
   all eight of these cases because I got it wrong the first time.

2. **The retry feature changed which demo queries are honest.** My planned demo of the
   size-retry branch was `"90s track jacket in size XS"` — but the One Size rule meant a bucket hat
   matched on the `90s` tag and the retry never fired. The behavior is right (a one-size item *does*
   fit an XS request); the demo query was wrong. I swapped in `"baggy carpenter jeans size S"` and
   `"corduroy wide-leg pants under $10"`, which genuinely exercise the two retry levels, and both
   are now examples in the UI and assertions in the test suite.

3. **The model in the spec no longer exists.** Covered under
   [A note on the model](#a-note-on-the-model) — this pushed a fallback chain into `_call_llm()`
   that the spec never contemplated. It also surfaced a second bug: the substitute models are
   *reasoning* models that spend part of the token budget thinking, so my original
   `max_tokens=200` returned empty content for the fit card. The guard I'd written for "LLM returns
   empty" fired correctly and told me exactly what was happening — an error path earning its keep
   during development, not just in the demo.

---

## AI Usage

I used **Claude (via Claude Code in VS Code)** throughout, driving it from the `planning.md` spec
rather than from loose descriptions.

**Instance 1 — implementing `search_listings` from the Tool 1 spec block.** I gave Claude the Tool 1
section (all three parameters with types, the return-value field list, and the "returns `[]`, never
raises" failure mode) plus `utils/data_loader.py`, and told it to use `load_listings()` rather than
re-open the JSON. What came back filtered on all three parameters and handled the empty case, but
matched sizes with a plain substring test. **I overrode that**: I had it replace the check with
token-based matching plus an alias map after working out that `"S" in "US 8"` is `True`, which would
silently return shoes for a shirt query. I also added the `"One Size"` wildcard, which neither the
spec nor the generated code had, and wrote the parametrized test with the `("S", "US 8", False)`
case specifically to pin the bug that had slipped through.

**Instance 2 — implementing the planning loop from the architecture diagram.** I gave Claude the
ASCII diagram from `planning.md` verbatim along with the Planning Loop and State Management
sections, and asked it to implement `run_agent()` against the existing `_new_session()` shape. The
diagram's explicit error branches meant the generated code got the early returns right the first
time. **What I changed:** the generated version treated a `create_fit_card` failure the same as a
search failure — blanking the whole session and returning only an error. That contradicted my own
error-handling table, which says a failed caption should keep the outfit. I rewrote step 6 to set
`error` while preserving `outfit_suggestion`, and rewrote `handle_query()` to fill panels
independently so a partial run still shows what worked. I also added the `tool_calls` log, which
wasn't in the spec — I wanted the loop's actual path inspectable rather than inferred, and it's what
let me catch that my size-retry demo query wasn't triggering the retry.

**Instance 3 — diagnosing the empty fit card.** When `create_fit_card` started returning
`[tool_error] The caption model returned an empty response`, I had Claude probe the API directly at
three different `max_tokens` values rather than guess. That showed the fallback models emit
reasoning tokens before content, so a short prompt fit in 200 tokens but my longer caption prompt
didn't. The fix was a budget raise (700 for captions, 900 for outfits), not a prompt change — which
is not where I would have looked first.

---

## What to Show in the Demo

Shot list for the 3–5 minute recording:

1. **Complete multi-step interaction** — `python app.py`, query *"vintage graphic tee under $30"*
   with the example wardrobe. Narrate each tool as its panel fills: search → price check → outfit →
   fit card.
2. **State passing** — run `python agent.py` and point at
   `[state] selected_item is search_results[0]: True`, plus the `Tools called:` log. Say out loud
   that the outfit step got the same object search returned, and the user never re-typed the item.
3. **Triggered failure** — submit *"designer ballgown size XXS under $5"*. Show the single error
   panel and read the message. Point out the other two panels stayed empty because
   `suggest_outfit` was never called.
4. **Retry fallback (stretch)** — submit *"baggy carpenter jeans size S"* and show the
   *"No listings in size S, so I searched every size instead"* note, then the run completing anyway.

---

## Project Structure

```
├── agent.py              # planning loop, query parsing, session state
├── tools.py              # the 4 tools + Groq client with model fallback
├── app.py                # Gradio UI, session → panel mapping
├── planning.md           # the spec, written before implementation
├── pytest.ini            # registers the `llm` marker
├── data/
│   ├── listings.json         # 40 mock listings
│   └── wardrobe_schema.json  # wardrobe schema + example/empty wardrobes
├── tests/
│   ├── test_tools.py     # tool behavior and failure modes
│   └── test_agent.py     # loop branching and state passing
└── utils/data_loader.py  # provided data helpers
```
