"""
rag_system.py
===========================================================================
Retrieval-Augmented Generation over the Economic Times corpus.

    1. Retrieve the top 5 articles with semantic_search.
    2. Build an evidence block from their title, topic, date and synopsis.
    3. Send the question + evidence to `mistral:latest` through Ollama.
    4. Return the answer together with the supporting ET sources.

Evidence rules enforced by the system prompt
    - Answer only from the supplied evidence.
    - No outside knowledge, no invented facts.
    - Say so plainly when the evidence does not cover the question.
    - Keep it clear and concise.
    - Never put a URL in the answer body (the sources are listed separately).
    - Cite with [Source N] so every claim is traceable to a link.

Two deliberate defences beyond the prompt, because a 7B model does not always
obey instructions:
    - any URL that still appears in the generated text is stripped out;
    - generation is capped (num_predict) and temperature is low, so a local model
      on a 4 GB laptop cannot ramble for minutes.

Usage
-----
    from rag_system import answer_question
    answer, sources = answer_question("How is AI affecting Indian companies?")
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config
from semantic_search import SearchError, search

URL_RE = re.compile(r"https?://\S+|www\.\S+")

SYSTEM_PROMPT = """You are an analyst working for a business newspaper. You answer \
questions using ONLY the Economic Times evidence supplied below.

Rules you must follow:
1. Use only the supplied evidence. Never use outside knowledge or general training \
knowledge about the companies, people or events mentioned.
2. Never invent, guess or extrapolate facts. If the evidence does not state something, \
it is not in your answer.
3. If the evidence does not answer the question, reply exactly: \
"The available Economic Times evidence does not answer this question." Then add one \
sentence on what the evidence does cover. Do not speculate.
4. Cite every claim inline with [Source N], matching the numbered evidence blocks.
5. Be clear and concise: short paragraphs or bullet points, no preamble, no closing \
pleasantries.
6. Never write a URL, link or bare domain in the answer. The sources are shown to the \
user separately, so refer to them only as [Source N].
7. Distinguish clearly between what the articles report and any hedging they contain. \
Do not upgrade a possibility into a fact."""



@dataclass
class Source:
    """
    One retrieved article, as shown under the answer.

    Carries the full six-field record from the dataset:
    Topic, Title, Article_URL, Publication_Date, Extraction_Date, Synopsis.
    """

    rank: int
    topic: str
    title: str
    article_url: str
    publication_date: str
    extraction_date: str
    synopsis: str
    score: float

    def to_dict(self) -> dict:
        """The six documented fields, plus rank and similarity score."""
        return {
            "rank": self.rank,
            "Topic": self.topic,
            "Title": self.title,
            "Article_URL": self.article_url,
            "Publication_Date": self.publication_date,
            "Extraction_Date": self.extraction_date,
            "Synopsis": self.synopsis,
            "score": self.score,
        }

    def to_row(self) -> dict:
        """Only the six documented dataset fields."""
        return {
            "Topic": self.topic,
            "Title": self.title,
            "Article_URL": self.article_url,
            "Publication_Date": self.publication_date,
            "Extraction_Date": self.extraction_date,
            "Synopsis": self.synopsis,
        }


@dataclass
class RagResult:
    """The answer plus the full records of the articles behind it."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    retrieved: int = 0
    refused: bool = False
    elapsed_seconds: float = 0.0

    @property
    def source_dicts(self) -> list[dict]:
        return [s.to_dict() for s in self.sources]

    @property
    def source_rows(self) -> list[dict]:
        """Only the six documented fields, ready for a DataFrame."""
        return [s.to_row() for s in self.sources]

class RagError(RuntimeError):
    """Raised when the answer cannot be produced at all."""


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def build_evidence(sources: list[Source]) -> str:
    """
    Render the retrieved articles as a numbered evidence block.

    Synopses are truncated to config.MAX_CTX_CHARS so five long articles cannot
    blow past the model's context or slow generation to a crawl.
    """
    blocks = []
    for src in sources:
        synopsis = (src.synopsis or "").strip()[: config.MAX_CTX_CHARS]
        blocks.append(
            f"[Source {src.rank}]\n"
            f"Title: {src.title}\n"
            f"Topic: {src.topic}\n"
            f"Published: {src.publication_date}\n"
            f"Summary text: {synopsis or '(no summary available)'}"
        )
    return "\n\n---\n\n".join(blocks)


def _to_sources(hits: pd.DataFrame) -> list[Source]:
    def text(value, limit=0):
        s = "" if value is None else str(value)
        if value is not None and not isinstance(value, str) and pd.isna(value):
            s = ""
        s = " ".join(s.split())
        return s[:limit] if limit else s

    return [
        Source(
            rank=int(row["rank"]),
            topic=text(row["Topic"]),
            title=text(row["Title"]),
            article_url=text(row["Article_URL"]),
            publication_date=text(row["Publication_Date"]),
            extraction_date=text(row["Extraction_Date"]),
            synopsis=text(row["Synopsis"]),
            score=float(row["score"]),
        )
        for _, row in hits.iterrows()
    ]


REFUSAL_MARKER = "does not answer this question"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(question: str, sources: list[Source]) -> tuple[str, float]:
    """
    Ask the local LLM for an answer grounded in `sources`.

    Returns (answer, seconds). Raises RagError if Ollama is unreachable.
    """
    import time

    import ollama

    evidence = build_evidence(sources)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"EVIDENCE\n{evidence}\n\n"
        f"QUESTION\n{question}\n\n"
        f"Write the answer now. Cite with [Source N]."
    )

    started = time.time()
    try:
        response = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                # A 7B model on a 4 GB GPU spills to system RAM; an unbounded
                # token budget is the difference between 20 s and 5 minutes.
                "num_predict": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096,
            },
        )
    except Exception as exc:                           # noqa: BLE001
        raise RagError(
            f"Could not reach {config.LLM_MODEL} through Ollama ({exc}).\n"
            f"       Is `ollama serve` running and the model pulled?\n"
            f"       Check with:  ollama list"
        ) from exc

    answer = (response.get("message", {}).get("content") or "").strip()

    # Belt-and-braces for rule 6: a local model will sometimes emit a URL even
    # when told not to.
    if URL_RE.search(answer):
        answer = URL_RE.sub("[link in sources]", answer)

    return answer, time.time() - started


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def answer_question(question: str,
                    top_k: int = config.TOP_K,
                    topic: str | None = None,
                    start: str | None = None,
                    end: str | None = None,
                    use_instruction: bool = True) -> RagResult:
    """
    Full RAG pass: retrieve, build evidence, generate, return answer + sources.
    """
    import time
    overall = time.time()

    question = (question or "").strip()
    if not question:
        raise RagError("Please enter a question.")

    hits = search(question, top_k=top_k, topic=topic, start=start, end=end,
                  use_instruction=use_instruction)
    sources = _to_sources(hits)
    if not sources:
        return RagResult(
            answer="No matching Economic Times articles were found for that question.",
            sources=[], retrieved=0, refused=True,
            elapsed_seconds=time.time() - overall,
        )

    answer, _ = generate(question, sources)
    if not answer:
        answer = "The available Economic Times evidence does not answer this question."

    return RagResult(
        answer=answer,
        sources=sources,
        retrieved=len(sources),
        refused=REFUSAL_MARKER in answer.lower(),
        elapsed_seconds=time.time() - overall,
    )


def answer_question_safe(question: str, **kwargs) -> tuple[RagResult | None, str | None]:
    """Non-raising variant for the UI: returns (result, error_message)."""
    try:
        return answer_question(question, **kwargs), None
    except (SearchError, RagError) as exc:
        return None, str(exc)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ask a question of the ET corpus (RAG)")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--topic", default="")
    args = ap.parse_args()

    question = " ".join(args.question)
    try:
        result = answer_question(question, top_k=args.top_k, topic=args.topic or None)
    except (SearchError, RagError) as exc:
        print(f"[rag] {exc}")
        raise SystemExit(1)

    print(f"\n{'=' * 78}\nQUESTION: {question}\n{'=' * 78}\n")
    print(result.answer)
    print(f"\n{'=' * 78}\nSOURCES ({result.retrieved})  [{result.elapsed_seconds:.1f}s total]\n{'=' * 78}")
    for s in result.sources:
        print(f"[{s.rank}] score {s.score:.3f}")
        for label, value in s.to_row().items():
            print(f"    {label:<17}: {value}")
        print()
