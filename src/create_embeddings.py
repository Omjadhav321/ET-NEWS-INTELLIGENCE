"""
create_embeddings.py
===========================================================================
Step 3 of the data pipeline.

Reads `Data/articles_clean.csv` and turns every article into a 1,024-dimension
vector using the local Ollama model `qwen3-embedding:0.6b`, writing:

    models/article_embeddings.npy     (22354, 1024) float32, row i == CSV row i
    models/embedding_meta.json        model, dim, counts, timing

Why precompute at all?
    Embedding 22,354 articles on demand would mean minutes per search. Only the
    *question* needs to be embedded at query time (~200 ms). The corpus is baked
    into a matrix once, here.

Design notes
    - Documents get NO task instruction; only queries do. See config.QUERY_INSTRUCTION.
    - Vectors are L2-normalised, so cosine similarity == dot product and search
      is a single matmul.
    - Ollama is asked for a whole *list* of texts per request. That is how it
      batches internally - far faster than one HTTP call per article, and it
      works even with OLLAMA_NUM_PARALLEL=1.
    - Progress is checkpointed and written atomically, so an interrupted run
      resumes with --resume instead of starting over.

Usage
-----
    python src/create_embeddings.py                      # full corpus, ~11 min
    python src/create_embeddings.py --bench 128          # throughput probe, writes nothing
    python src/create_embeddings.py --limit 200          # quick test build
    python src/create_embeddings.py --resume             # continue after a crash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Ollama plumbing
# ---------------------------------------------------------------------------
def ollama_available() -> bool:
    """True if the local Ollama HTTP server answers."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:                                  # noqa: BLE001
        return False


def require_model(name: str) -> None:
    """Fail loudly and usefully if the embedding model is not pulled."""
    import ollama
    pulled = {m["model"] for m in ollama.list().get("models", [])}
    if not any(p == name or p.startswith(name.split(":")[0]) for p in pulled):
        raise SystemExit(
            f"[embed] model '{name}' is not installed.\n"
            f"        Run:  ollama pull {name}\n"
            f"        (and make sure `ollama serve` is running in another terminal)"
        )


def embed_batch(texts: list[str], model: str) -> np.ndarray:
    """
    One Ollama call for a whole batch -> (len(texts), dim) float32, unit-norm.

    Retries with backoff, because a laptop switching Wi-Fi access points
    mid-run is the most common way a long build dies.
    """
    import ollama

    for attempt in range(4):
        try:
            response = ollama.embed(model=model, input=texts, truncate=True)
            vectors = np.asarray(response["embeddings"], dtype=np.float32)
            break
        except Exception as exc:                       # noqa: BLE001
            if attempt == 3:
                raise SystemExit(f"[embed] Ollama embedding failed after 4 tries: {exc}") from exc
            wait = 2 ** attempt
            print(f"\n[embed]   retry {attempt + 1}/4 after {wait}s: {exc}", flush=True)
            time.sleep(wait)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[0] != len(texts):
        raise SystemExit(f"[embed] Ollama returned {vectors.shape[0]} vectors for {len(texts)} inputs")

    # L2 normalise so cosine similarity == plain dot product at query time.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return vectors / norms


def human(n: float) -> str:
    return f"{n / 1048576:.1f} MB"


def eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------
def select(df: pd.DataFrame, args) -> pd.DataFrame:
    subset = df
    if args.topic:
        wanted = {t.strip().lower() for t in args.topic.split(",")}
        subset = subset[subset["Topic"].str.lower().isin(wanted)]
    if args.start or args.end:
        parsed = config.parse_publication_series(subset["Publication_Date"])
        if args.start:
            subset = subset[parsed >= pd.Timestamp(args.start)]
        if args.end:
            subset = subset[parsed <= pd.Timestamp(args.end) + pd.Timedelta(days=1)]
    if args.limit and args.limit > 0:
        subset = subset.head(args.limit)
    return subset.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _checkpoint_paths() -> list[tuple[Path, Path]]:
    """
    Candidate (checkpoint, indices) pairs, best first.

    The cache dir is preferred because the checkpoint is 87 MB and is rewritten
    every few hundred articles; this project lives in a OneDrive folder, where
    large repeated writes are slow and can hit file locks. `models/` stays as a
    fallback so an older checkpoint still resumes.
    """
    return [
        (config.CHECKPOINT_DIR / "embedding_checkpoint.npy",
         config.CHECKPOINT_DIR / "embedding_indices.npy"),
        (config.CHECKPOINT_NPY, config.INDICES_NPY),
    ]


def load_checkpoint(total: int, dim: int) -> tuple[np.ndarray, int]:
    """Return (partial_matrix, rows_done), or a zero matrix if there is nothing usable."""
    matrix = np.zeros((total, dim), dtype=np.float32)
    for ckpt_path, idx_path in _checkpoint_paths():
        if not ckpt_path.exists() or not idx_path.exists():
            continue
        try:
            partial = np.load(ckpt_path)
            indices = np.load(idx_path)
            if partial.shape != matrix.shape:
                print(f"[embed] ignoring checkpoint: shape {partial.shape} != {matrix.shape}")
                continue
            done = int(indices.size)
            matrix[indices[:done]] = partial[indices[:done]]
            print(f"[embed] resuming from checkpoint at row {done:,}/{total:,}  ({ckpt_path})")
            return matrix, done
        except Exception as exc:                       # noqa: BLE001
            print(f"[embed] ignoring unreadable checkpoint: {exc}")
    return matrix, 0


def save_checkpoint(matrix: np.ndarray, rows_done: int) -> None:
    """
    Persist rows 0..rows_done-1.

    The indices file is written first: a checkpoint without its indices is
    correctly ignored by load_checkpoint, whereas indices pointing at a missing
    matrix would be ignored too - so the pair can never be half-trusted.
    """
    ckpt_path, idx_path = _checkpoint_paths()[0]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(idx_path, np.arange(rows_done, dtype=np.int64))

    # The temp name must still end in ".npy" - numpy appends the extension when
    # it is missing, which would leave tmp.replace() pointing at a file that was
    # never written.
    tmp = ckpt_path.with_name(f"{ckpt_path.stem}.part.npy")
    np.save(tmp, matrix)
    tmp.replace(ckpt_path)                             # atomic on Windows


def clear_checkpoint() -> None:
    for ckpt_path, idx_path in _checkpoint_paths():
        ckpt_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute Qwen3 embeddings for the ET corpus")
    ap.add_argument("--articles", default=str(config.ARTICLES_CSV))
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--out", default=str(config.EMBEDDINGS_NPY))
    ap.add_argument("--batch-size", type=int, default=128,
                    help="texts per Ollama request (measured on an RTX 3050: "
                         "64 -> 25 art/s, 128 -> 37 art/s, 256 -> 10 art/s)")
    ap.add_argument("--limit", type=int, default=0, help="only embed the first N articles")
    ap.add_argument("--topic", default="", help="comma-separated topic filter")
    ap.add_argument("--start", default="", help="earliest Publication_Date (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="latest Publication_Date (YYYY-MM-DD)")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="time N articles and exit, writing nothing")
    args = ap.parse_args()

    config.ensure_dirs()

    # ---- preflight --------------------------------------------------------
    articles_path = Path(args.articles)
    if not articles_path.exists():
        print(f"[embed] {articles_path} not found. Run python src/clean_data.py first", file=sys.stderr)
        return 1
    if not ollama_available():
        print(f"[embed] cannot reach Ollama at {config.OLLAMA_HOST}", file=sys.stderr)
        print("        Start it with:  ollama serve", file=sys.stderr)
        return 1
    require_model(args.model)

    df = pd.read_csv(articles_path, dtype=config.CSV_DTYPES, keep_default_na=False, na_values=[""])
    subset = select(df, args)
    texts = [config.document_text(t, s, p)
             for t, s, p in zip(subset["Title"], subset["Synopsis"], subset["Topic"])]
    total = len(texts)

    print(f"[embed] corpus   : {len(df):,} articles -> embedding {total:,}")
    print(f"[embed] model    : {args.model}")
    print(f"[embed] batch    : {args.batch_size} texts/request")
    print(f"[embed] text     : title + synopsis (documents carry no task instruction)")

    # ---- benchmark mode ---------------------------------------------------
    if args.bench:
        probe = texts[: args.bench]
        t0 = time.time()
        vectors = embed_batch(probe, args.model)
        dt = time.time() - t0
        print(f"[embed] {args.bench} articles in {dt:.1f}s = {args.bench / dt:.1f} art/s"
              f"  |  full corpus would take ~{eta(total * dt / args.bench)}")
        print(f"[embed] vector dim {vectors.shape[1]} (expected {config.EMBED_DIM})")
        print(f"[embed] sample norms {[round(float(x), 4) for x in np.linalg.norm(vectors, axis=1)[:3]]}")
        return 0

    if total == 0:
        print("[embed] selection is empty - nothing to do")
        return 1

    # ---- resume -----------------------------------------------------------
    if args.resume:
        matrix, start_row = load_checkpoint(total, config.EMBED_DIM)
    else:
        matrix, start_row = np.zeros((total, config.EMBED_DIM), dtype=np.float32), 0
        clear_checkpoint()

    # ---- main loop --------------------------------------------------------
    started = time.time()
    print(f"\n{'progress':>30}   {'rate':>13}   {'eta':>10}")
    try:
        for i in range(start_row, total, args.batch_size):
            batch = texts[i : i + args.batch_size]
            matrix[i : i + len(batch)] = embed_batch(batch, args.model)
            done = min(i + len(batch), total)

            if done % (args.batch_size * 4) == 0 or done == total:
                save_checkpoint(matrix, done)

            elapsed = max(time.time() - started, 1e-9)
            rate = (done - start_row) / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            filled = int(24 * done / total)
            print(f"\r[embed] {'#' * filled}{'.' * (24 - filled)} {done:>7,}/{total:,}   "
                  f"{rate:>8.1f} art/s   {eta(remaining):>10}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[embed] interrupted - checkpoint saved, rerun with --resume to continue")
        return 130
    print()

    compute = time.time() - started
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    # ---- validation -------------------------------------------------------
    problems = []
    norms = np.linalg.norm(matrix[: min(64, total)], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        problems.append(f"vectors are not unit-norm (sample {[round(float(x), 4) for x in norms[:3]]})")
    if np.isnan(matrix).any():
        problems.append("NaNs in the output matrix")
    if total == len(df) and total != 22354:
        problems.append(f"expected the full 22,354-article corpus but selected {total}")

    if problems:
        print("\n[embed] FAILED validation:", file=sys.stderr)
        for p in problems:
            print(f"        - {p}", file=sys.stderr)
        return 1

    # ---- write ------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, matrix)

    meta = {
        "model": args.model,
        "dim": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "normalize": True,
        "pooling": "model default (last-token for Qwen3-Embedding)",
        "text": "title + synopsis",
        "queryInstruction": config.QUERY_INSTRUCTION,
        "documentsCarryInstruction": False,
        "articlesCsv": articles_path.name,
        "totalArticlesInCsv": int(len(df)),
        "isFullCorpus": bool(total == len(df)),
        "selection": {
            "limit": args.limit or None,
            "topic": args.topic or None,
            "start": args.start or None,
            "end": args.end or None,
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "computeSeconds": round(compute, 1),
        "articlesPerSecond": round(total / compute, 2) if compute else None,
        "bytes": out_path.stat().st_size,
    }
    config.EMBEDDING_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    clear_checkpoint()

    print("\n[embed] -------------------------------------------------------")
    print(f"[embed] vectors    : {matrix.shape[0]:,} x {matrix.shape[1]}  ({human(out_path.stat().st_size)})")
    print(f"[embed] throughput : {meta['articlesPerSecond']} art/s  ({compute:.0f}s total)")
    print(f"[embed] metadata   : {config.EMBEDDING_META.name}")
    print("[embed] -------------------------------------------------------")

    # Nearest-neighbour probe: proves the vectors encode meaning rather than
    # just being unit-norm noise.
    probe_idx = next((i for i, t in enumerate(subset["Title"]) if "Adani Green" in str(t)), 0)
    sims = matrix @ matrix[probe_idx]
    print(f"[embed] sanity check - nearest neighbours of:")
    print(f"           \"{subset['Title'].iloc[probe_idx]}\"")
    for rank in np.argsort(-sims)[:4]:
        print(f"           {sims[rank]:.3f}  {subset['Title'].iloc[int(rank)][:68]}")
    print("[embed] Next: python src/test_classification.py   then   streamlit run app/app.py")
    print("[embed] -------------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())