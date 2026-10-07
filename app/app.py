"""
app.py
===========================================================================
ET News Intelligence - Streamlit front end.

Two views, matching the project documentation:

    Ask Question      a natural-language question is embedded, the top 5 articles
                      are retrieved, and mistral:latest writes an answer using only
                      that evidence. The supporting ET articles are listed below
                      the answer with titles, metadata, links and match scores.

    Explore News      browse the same corpus by topic and date range, newest
                      first, with links back to the original articles.

Design: white background, black text, one bold blue accent, large bold
headings, editorial layout. No sidebar, no gradients, no chat bubbles.

Run
---
    streamlit run app/app.py
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

import config
from semantic_search import SearchError, available_topics, index_info, load_corpus, search
from rag_system import answer_question_safe

CSS_PATH = ROOT / "app" / "assets" / "style.css"

st.set_page_config(
    page_title="ET News Intelligence",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_index() -> dict:
    """Load and validate the corpus + embedding matrix once per session."""
    return index_info()


@st.cache_data(show_spinner=False)
def get_corpus_snapshot() -> pd.DataFrame:
    return load_corpus()


@st.cache_data(show_spinner=False)
def run_search(question: str, top_k: int, topic: str | None) -> pd.DataFrame:
    return search(question, top_k=top_k, topic=topic)


@st.cache_data(show_spinner=False)
def run_rag(question: str, top_k: int) -> tuple[dict, list[dict], bool, float]:
    result, error = answer_question_safe(question, top_k=top_k)
    if error:
        return {"answer": error}, [], False, 0.0
    return ({"answer": result.answer}, result.source_dicts,
            result.refused, result.elapsed_seconds)


def ollama_status() -> tuple[bool, str]:
    try:
        import ollama
        models = {m["model"] for m in ollama.list().get("models", [])}
    except Exception as exc:                            # noqa: BLE001
        return False, f"cannot reach Ollama ({exc})"

    have_embed = any(m == config.EMBED_MODEL or m.startswith(config.EMBED_MODEL.split(":")[0])
                     for m in models)
    have_llm = any(m == config.LLM_MODEL or m.startswith(config.LLM_MODEL.split(":")[0])
                   for m in models)
    if have_embed and have_llm:
        return True, "ready"
    missing = [n for n, ok in ((config.EMBED_MODEL, have_embed), (config.LLM_MODEL, have_llm)) if not ok]
    return False, "not pulled: " + ", ".join(missing)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def esc(value) -> str:
    if value is None:
        return ""
    try:
        if not isinstance(value, str) and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).split())
    for ch, rep in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")):
        text = text.replace(ch, rep)
    return text


def render_article(record: dict, score: float | None = None, show_url: bool = True) -> None:
    """
    One newspaper-style story row.

    `record` holds the six documented dataset fields:
    Topic, Title, Article_URL, Publication_Date, Extraction_Date, Synopsis.
    """
    topic = esc(record.get("Topic"))
    title = esc(record.get("Title"))
    pub = esc(record.get("Publication_Date"))
    extracted = esc(record.get("Extraction_Date"))
    url = esc(record.get("Article_URL"))
    synopsis = esc(record.get("Synopsis"))

    score_html = f'<span class="et-score">{score:.3f}</span>' if score is not None else ""
    extracted_html = f" &nbsp;|&nbsp; extracted {extracted}" if extracted else ""
    url_html = (
        f'<div class="et-item-url">{url}</div>' if (show_url and url) else ""
    )

    st.markdown(
        f'<div class="et-item">'
        f'<div class="et-item-meta">{score_html}{topic} &nbsp;|&nbsp; {pub}{extracted_html}</div>'
        f'<div class="et-item-title">'
        f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
        f"</div>"
        f'<p class="et-item-synopsis">{synopsis}</p>'
        f"{url_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_answer(text: str) -> None:
    st.markdown(f'<div class="et-answer">{answer_to_html(text)}</div>',
                unsafe_allow_html=True)


# The answer is dropped inside a styled <div>, and block-level raw HTML switches
# markdown parsing off for everything nested inside it. Feeding the model's
# markdown straight through therefore flattened every numbered list and bold run
# into one run-on paragraph, so the handful of constructs it actually emits are
# converted here instead.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", re.S)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET_RE = re.compile(r"^\s{0,3}[-*•]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s{0,3}\d{1,2}[.)]\s+(.*)$")


def _escape_keep_lines(text: str) -> str:
    """Like `esc`, but without collapsing newlines - the answer is multi-line."""
    for ch, rep in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;")):
        text = text.replace(ch, rep)
    return text


def _inline(text: str) -> str:
    """Escape one line, then re-apply the inline marks the answer model uses."""
    out = _escape_keep_lines(text.strip())
    out = _BOLD_RE.sub(r"<strong>\1</strong>", out)
    out = _ITALIC_RE.sub(r"<em>\1</em>", out)
    return out


def answer_to_html(text: str) -> str:
    """Render the generated answer as HTML that keeps its paragraphs and lists."""
    blocks: list[str] = []
    para: list[str] = []
    items: list[str] = []
    list_tag: str | None = None

    def close_list() -> None:
        nonlocal list_tag, items
        if list_tag is not None:
            blocks.append(f"<{list_tag}>{''.join(items)}</{list_tag}>")
        list_tag, items = None, []

    def close_para() -> None:
        if para:
            blocks.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    for line in (text or "").replace("\r\n", "\n").split("\n"):
        if not line.strip():
            close_list()
            close_para()
            continue

        heading = _HEADING_RE.match(line)
        bullet = _BULLET_RE.match(line)
        numbered = _NUMBERED_RE.match(line)

        if heading:
            close_list()
            close_para()
            blocks.append(f"<p><strong>{_inline(heading.group(1))}</strong></p>")
        elif bullet:
            close_para()
            if list_tag != "ul":
                close_list()
                list_tag = "ul"
            items.append(f"<li>{_inline(bullet.group(1))}</li>")
        elif numbered:
            close_para()
            if list_tag != "ol":
                close_list()
                list_tag = "ol"
            items.append(f"<li>{_inline(numbered.group(1))}</li>")
        else:
            close_list()
            para.append(line.strip())

    close_list()
    close_para()
    return "".join(blocks) or "<p></p>"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

try:
    info = get_index()
    index_ok = True
except (SearchError, FileNotFoundError) as exc:
    info = {}
    index_ok = False
    index_error = str(exc)

llm_ok, llm_note = ollama_status() if index_ok else (False, "index not built")

if index_ok:
    st.markdown(
        '<div class="et-masthead">'
        '<p class="et-kicker">Economic Times &nbsp;&middot;&nbsp; Retrieval-Augmented Generation</p>'
        "<h1>ET News Intelligence</h1>"
        '<p class="et-standfirst">Ask a question in plain English and get an answer built '
        "only from retrieved Economic Times coverage &mdash; with every claim traced back "
        "to the article it came from.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="et-dateline">{info["articles"]:,} articles &nbsp;|&nbsp; '
        f'{info["oldest"]} to {info["newest"]} &nbsp;|&nbsp; {info["topics"]} topics &nbsp;|&nbsp; '
        f'{info["dim"]}-dim embeddings by {esc(info["model"])}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="et-masthead"><p class="et-kicker">Economic Times</p>'
        "<h1>ET News Intelligence</h1></div>",
        unsafe_allow_html=True,
    )
    st.error(index_error)

if index_ok and not llm_ok:
    st.warning(
        f"Ollama is reachable but not fully set up ({llm_note}). "
        f"Run `ollama serve`, then `ollama pull {config.EMBED_MODEL}` and "
        f"`ollama pull {config.LLM_MODEL}`. Semantic search needs the embedding model; "
        "written answers need the language model.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
if index_ok:
    ask_tab, explore_tab = st.tabs(["Ask Question", "Explore News by Time"])

    # ---- Ask Question ---------------------------------------------------
    with ask_tab:
        st.markdown("## Ask a question")
        st.markdown(
            "The question is embedded with the same model that indexed the corpus, "
            f"the {config.TOP_K} closest articles become the evidence, and "
            f"`{config.LLM_MODEL}` writes the answer using that evidence only."
        )

        with st.form("ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="How is AI affecting Indian companies?",
                label_visibility="collapsed",
            )
            top_k = st.slider("Articles used as evidence", 3, 10, config.TOP_K)
            submitted = st.form_submit_button("Search the archive")

        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                started = time.time()
                with st.spinner("Retrieving articles and writing the answer..."):
                    answer, sources, refused, elapsed = run_rag(question.strip(), top_k)

                if answer.get("answer", "").startswith(("Could not", "Please enter")):
                    st.error(answer["answer"])
                else:
                    if refused:
                        st.info("The model judged the retrieved evidence insufficient.")

                    render_answer(answer["answer"])

                    st.markdown(f"### Sources ({len(sources)})")
                    st.markdown(
                        '<div class="et-note">Match score is cosine similarity between the '
                        "question and the article, 1.00 being an exact match. "
                        "Titles link to the original Economic Times article.</div>",
                        unsafe_allow_html=True,
                    )
                    for src in sources:
                        render_article(src, score=src.get("score"))
                    st.markdown(
                        f'<div class="et-note">Answered in {elapsed:.1f}s '
                        f"({len(sources)} articles retrieved). Models run locally; "
                        "expect several seconds on a laptop.</div>",
                        unsafe_allow_html=True,
                    )

    # ---- Explore News ---------------------------------------------------
    with explore_tab:
        st.markdown("## Explore news by time")
        st.markdown("Browse the same archive by topic and date range, newest first.")

        corpus = get_corpus_snapshot()
        topics = available_topics()
        dates = corpus["publication_dt"]

        col1, col2, col3 = st.columns([2, 1.4, 1.4])
        with col1:
            topic_choice = st.selectbox("Topic", ["All topics"] + topics)
        with col2:
            start_date = st.date_input(
                "From", value=dates.min().date() if not dates.isna().all() else date(2021, 1, 1)
            )
        with col3:
            end_date = st.date_input(
                "To", value=dates.max().date() if not dates.isna().all() else date(2026, 7, 20)
            )

        keyword = st.text_input("Filter headlines containing...", placeholder="optional")

        if start_date > end_date:
            st.warning("The start date is after the end date - showing nothing.")
            matches = corpus.iloc[0:0]
        else:
            mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
            if topic_choice != "All topics":
                mask &= corpus["Topic"] == topic_choice
            if keyword.strip():
                needle = keyword.strip().lower()
                mask &= corpus["Title"].str.lower().str.contains(needle, regex=False, na=False)
            matches = corpus[mask].sort_values("publication_dt", ascending=False)

        st.markdown(
            f"### {len(matches):,} article{'s' if len(matches) != 1 else ''}"
            + (f" in {topic_choice}" if topic_choice != "All topics" else "")
            + (f' matching "{keyword.strip()}"' if keyword.strip() else "")
        )

        if matches.empty:
            st.info("No articles match those filters.")
        else:
            page_size = 20
            pages = max(1, -(-len(matches) // page_size))
            page = 1
            if pages > 1:
                nav = st.columns([1, 3, 1])
                with nav[0]:
                    if st.button("← Newer", disabled=page == 1):
                        st.session_state["explore_page"] = max(1, st.session_state.get("explore_page", 1) - 1)
                with nav[1]:
                    st.markdown(
                        f'<div class="et-note" style="text-align:center;border:none;padding-top:0.5rem">'
                        f"Page {st.session_state.get('explore_page', 1)} of {pages}</div>",
                        unsafe_allow_html=True,
                    )
                with nav[2]:
                    if st.button("Older →", disabled=st.session_state.get("explore_page", 1) >= pages):
                        st.session_state["explore_page"] = min(pages, st.session_state.get("explore_page", 1) + 1)
                page = st.session_state.get("explore_page", 1)

            window = matches.iloc[(page - 1) * page_size : page * page_size]
            for _, row in window.iterrows():
                render_article(row.to_dict())

# ---------------------------------------------------------------------------
# Colophon
# ---------------------------------------------------------------------------
db_note = "CSV dataset"
try:
    import Database_Connection
    if Database_Connection.is_available():
        db_note = f"MySQL {config.MYSQL['host']}/{config.MYSQL['database']}"
except Exception:                                      # noqa: BLE001
    pass

st.markdown(
    f'<div class="et-colophon">'
    f"<strong>ET News Intelligence</strong> &nbsp;|&nbsp; Phase 1<br>"
    f"Embeddings: {esc(config.EMBED_MODEL)} ({config.EMBED_DIM} dims) &nbsp;|&nbsp; "
    f"Answers: {esc(config.LLM_MODEL)} &nbsp;|&nbsp; Retrieval: top {config.TOP_K} semantic<br>"
    f"Data: {db_note} &nbsp;|&nbsp; Everything runs locally; no article text is sent anywhere."
    f"</div>",
    unsafe_allow_html=True,
)