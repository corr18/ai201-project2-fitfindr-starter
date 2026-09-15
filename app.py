"""
app.py

Gradio interface for FitFindr. handle_query() calls run_agent() and maps the
returned session dict onto the three output panels — listing, outfit, fit card —
checking session["error"] first so a partial run still shows whatever succeeded.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def _format_listing(session: dict) -> str:
    """Render the selected listing, plus anything the agent decided about it."""
    item = session["selected_item"]
    lines = [
        item["title"],
        "",
        f"${item['price']:.0f} · {item['platform']} · size {item['size']}",
        f"Condition: {item['condition']}",
        f"Style: {', '.join(item['style_tags'])}",
        "",
        item["description"],
    ]

    price_check = session.get("price_check")
    if price_check and price_check["verdict"] != "not enough data":
        lines += ["", f"💰 Price check: {price_check['message']}"]

    # Notes explain anything the agent did that the user didn't ask for —
    # loosening a filter, or falling back to generic styling advice.
    if session.get("notes"):
        lines += [""] + [f"ℹ️ {note}" for note in session["notes"]]

    total = len(session.get("search_results") or [])
    if total > 1:
        lines += ["", f"(Top match of {total} — showing the best one.)"]

    return "\n".join(lines)


def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:      The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of three strings:
            (listing_text, outfit_suggestion, fit_card)
        Each string maps to one of the three output panels in the UI.
    """
    if not user_query or not user_query.strip():
        return (
            "Type what you're looking for first — e.g. "
            "\"vintage graphic tee under $30, size M\".",
            "",
            "",
        )

    wardrobe = (
        get_empty_wardrobe()
        if wardrobe_choice == "Empty wardrobe (new user)"
        else get_example_wardrobe()
    )

    session = run_agent(user_query, wardrobe)

    # An error before a listing was selected means there is nothing to show at
    # all — the error goes in the first panel and the others stay empty.
    if session["error"] and not session["selected_item"]:
        return f"⚠️ {session['error']}", "", ""

    listing_text = _format_listing(session)

    # A later step can fail after the listing was found. Keep every panel that
    # did succeed and put the error only in the panel that didn't.
    outfit_text = session["outfit_suggestion"] or f"⚠️ {session['error']}"
    if session["outfit_suggestion"]:
        fit_card_text = session["fit_card"] or f"⚠️ {session['error']}"
    else:
        fit_card_text = "Skipped — the outfit step needs to succeed first."

    return listing_text, outfit_text, fit_card_text


# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "baggy carpenter jeans size S",          # triggers the size-filter retry
    "corduroy wide-leg pants under $10",     # triggers the price-cap retry
    "designer ballgown size XXS under $5",   # deliberate no-results test
]

def build_interface():
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
