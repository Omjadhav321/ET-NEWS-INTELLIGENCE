"""
semantic_search.py
===========================================================================
Semantic retrieval over the precomputed article embeddings.

    1. The question is turned into one 1,024-dim vector via Ollama.
    2. It is dotted against every row of `models/article_embeddings.npy`.
    3. Because both sides are L2-normalised, that dot product *is* cosine
       similarity - no per-row normalisation needed at query time.
    4. The top 5 rows become the evidence for the RAG step.

The 91 MB matrix and the 22,354-row CSV are loaded once per process and cached.
Without that, a Streamlit rerun would re-read 105 MB on every keystroke.

Row alignment is verified on load: if the matrix row count no longer matches the
CSV, searching raises instead of silently returning mismatched articles.

Usage
-----
    from semantic_search import search
    hits = search("How is AI affecting Indian companies?")
    hits[["Title", "Topic", "score"]]
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config


class SearchError(RuntimeError):
    """Raised when the index is missing or cannot be trusted."""


# ---------------------------------------------------------------------------
# Loading (cached for the life of the process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_corpus() -> pd.DataFrame:
    """The article table, with a parsed publication date for filtering."""
    if not config.ARTICLES_CSV.exists():
        raise SearchError(
            f"{config.ARTICLES_CSV} not found. Run:\n"
            f"       python src/combine_data.py\n"
            f"       python src/clean_data.py"
        )
    df = pd.read_csv(config.ARTICLES_CSV, dtype=config.CSV_DTYPES,
                     keep_default_na=False, na_values=[""])
    df["publication_dt"] = config.parse_publication_series(df["Publication_Date"])
    return df


@lru_cache(maxsize=1)
def load_matrix() -> np.ndarray:
    """The embedding matrix, validated against the corpus row count."""
    if not config.EMBEDDINGS_NPY.exists():
        raise SearchError(
            f"{config.EMBEDDINGS_NPY} not found. Run:\n"
            f"       python src/create_embeddings.py"
        )
    matrix = np.load(config.EMBEDDINGS_NPY, mmap_mode="r")

    df = load_corpus()
    if matrix.shape[0] != len(df):
        raise SearchError(
            f"Index is stale: the matrix has {matrix.shape[0]:,} rows but the CSV has "
            f"{len(df):,}. Rebuild with:  python src/create_embeddings.py"
        )
    if matrix.shape[1] != config.EMBED_DIM:
        raise SearchError(f"Index has dim {matrix.shape[1]}, expected {config.EMBED_DIM}")
    return matrix


def index_info() -> dict:
    """Everything the UI shows in the footer / status line."""
    df = load_corpus()
    matrix = load_matrix()
    meta = {}
    if config.EMBEDDING_META.exists():
        import json
        meta = json.loads(config.EMBEDDING_META.read_text(encoding="utf-8"))
    dates = df["publication_dt"]
    return {
        "articles": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "model": meta.get("model", config.EMBED_MODEL),
        "builtAt": meta.get("generatedAt", "unknown"),
        "computeSeconds": meta.get("computeSeconds"),
        "topics": int(df["Topic"].nunique()),
        "oldest": None if dates.isna().all() else str(dates.min().date()),
        "newest": None if dates.isna().all() else str(dates.max().date()),
    }


def available_topics() -> list[str]:
    """Topic labels, most articles first - the Explore News dropdown order."""
    return load_corpus()["Topic"].value_counts().index.tolist()


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------
def embed_query(question: str, use_instruction: bool = True) -> np.ndarray:
    """
    One question -> one unit-norm vector.

    The instruction prefix matters: Qwen3-Embedding is trained so *queries* carry
    a task instruction while *documents* do not. See config.QUERY_INSTRUCTION.
    """
    text = config.format_query(question, use_instruction=use_instruction)
    if not text:
        raise SearchError("Empty question")

    import ollama
    try:
        response = ollama.embed(model=config.EMBED_MODEL, input=text, truncate=True)
    except Exception as exc:                           # noqa: BLE001
        raise SearchError(
            f"Could not embed the question via Ollama ({exc}).\n"
            f"       Is `ollama serve` running and `{config.EMBED_MODEL}` pulled?"
        ) from exc

    vec = np.asarray(response["embeddings"][0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        raise SearchError("The embedding model returned a zero vector")
    return vec / norm


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(question: str,
           top_k: int = config.TOP_K,
           topic: str | None = None,
           start: str | None = None,
           end: str | None = None,
           min_score: float | None = None,
           use_instruction: bool = True) -> pd.DataFrame:
    """
    Top-`top_k` articles for `question`.

    Optional topic / date filters are applied to the *candidates* before
    ranking, so filtering narrows the search rather than hiding results
    afterwards.

    Returns a DataFrame with the article columns plus `score` (cosine, 1.0 = the
    article is about exactly this) and `rank`.
    """
    df = load_corpus()
    matrix = load_matrix()

    query_vec = embed_query(question, use_instruction=use_instruction)
    if query_vec.shape[0] != matrix.shape[1]:
        raise SearchError(f"Query vector dim {query_vec.shape[0]} != index dim {matrix.shape[1]}")

    # Narrow the candidate set first when a filter is requested.
    candidates = np.arange(len(df))
    if topic:
        wanted = {t.strip().lower() for t in topic.split(",") if t.strip()}
        mask = df["Topic"].str.lower().isin(wanted).to_numpy()
        candidates = candidates[mask]
        if candidates.size == 0:
            raise SearchError(f"No articles in topic(s): {topic}")
    if start or end:
        dates = df["publication_dt"]
        keep = np.ones(len(df), dtype=bool)
        if start:
            keep &= (dates >= pd.Timestamp(start)).to_numpy()
        if end:
            keep &= (dates <= pd.Timestamp(end) + pd.Timedelta(days=1)).to_numpy()
        candidates = candidates[keep]
        if candidates.size == 0:
            raise SearchError(f"No articles between {start or 'the beginning'} and {end or 'today'}")

    # matmul over just the candidates: (n, dim) @ (dim,) -> (n,)
    scores = np.asarray(matrix[candidates]) @ query_vec

    k = min(top_k, scores.size)
    # argpartition finds the top k in O(n); a full sort of 22k is needless here.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]                # order just those k

    rows = df.iloc[candidates[top]].copy()
    rows["score"] = scores[top]
    rows["rank"] = np.arange(1, k + 1)

    if min_score is not None:
        rows = rows[rows["score"] >= min_score]
    return rows.reset_index(drop=True)


def search_batch(questions: list[str], top_k: int = config.TOP_K) -> dict[str, pd.DataFrame]:
    """Several questions, one query-embedding call per question, shared index."""
    return {q: search(q, top_k=top_k) for q in questions}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Search the ET article index")
    ap.add_argument("question", nargs="+", help="the question to ask")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--topic", default="", help="restrict to these topics")
    ap.add_argument("--start", default="", help="earliest Publication_Date (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="latest Publication_Date (YYYY-MM-DD)")
    ap.add_argument("--no-instruct", action="store_true",
                    help="embed the query without the Qwen3 task instruction")
    args = ap.parse_args()

    question = " ".join(args.question)
    try:
        info = index_info()
        print(f"[search] index: {info['articles']:,} articles x {info['dim']} dims "
              f"({info['model']}, built {info['builtAt']})")
        hits = search(question, top_k=args.top_k, topic=args.topic or None,
                      start=args.start or None, end=args.end or None,
                      use_instruction=not args.no_instruct)
    except SearchError as exc:
        print(f"[search] {exc}")
        raise SystemExit(1)

    print(f'[search] "{question}"\n')
    for _, row in hits.iterrows():
        print(f"  {row['rank']}. score {row['score']:.3f}")
        for field in config.DATASET_COLUMNS:
            print(f"       {field:<17}: {row[field]}")
        print()