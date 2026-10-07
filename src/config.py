"""
config.py
===========================================================================
Single source of truth for every path, model name and tunable in the project.

Nothing else in `src/` is allowed to hard-code a path or a model name. That
way the dataset and the embeddings can never drift apart silently, and swapping
the answer model is a one-line change.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "Data"
MODELS_DIR = ROOT / "models"
RAW_DIR = ROOT / "ET_Data_fetch.json"
RAW_JSON = ROOT / "data" / "articles.json"          # consolidated Node build (optional shortcut)

ARTICLES_RAW_CSV = DATA_DIR / "articles_raw.csv"
ARTICLES_CSV = DATA_DIR / "articles_clean.csv"

# The six documented fields, in order. Every retrieval result carries exactly
# these (plus a similarity score and rank), and the dataset is validated against
# them, so nothing downstream can silently drop a column.
DATASET_COLUMNS = [
    "Topic",
    "Title",
    "Article_URL",
    "Publication_Date",
    "Extraction_Date",
    "Synopsis",
]

EMBEDDINGS_NPY = MODELS_DIR / "article_embeddings.npy"
CHECKPOINT_NPY = MODELS_DIR / "embedding_checkpoint.npy"
INDICES_NPY = MODELS_DIR / "embedding_indices.npy"
EMBEDDING_META = MODELS_DIR / "embedding_meta.json"

CLEAN_REPORT = DATA_DIR / "cleaning_report.json"

# The checkpoint is written repeatedly during a multi-hour run. Writing ~90 MB
# into a OneDrive-backed folder that often is slow and can hit file locks, so
# it defaults to a local cache directory and only the finished matrix is copied
# into models/.
CHECKPOINT_DIR = Path(os.environ.get("ET_NEWS_CACHE", Path.home() / ".et_news_cache"))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("ET_EMBED_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL = os.environ.get("ET_LLM_MODEL", "mistral:latest")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

EMBED_DIM = 1024          # qwen3-embedding:0.6b hidden size
TOP_K = 5                 # documents handed to the LLM as evidence
MAX_CTX_CHARS = 512       # per-article evidence cap; keeps the prompt ~1.5k tokens

# ---------------------------------------------------------------------------
# Embedding text
# ---------------------------------------------------------------------------
# Qwen3-Embedding is instruction-tuned: *queries* carry a task instruction,
# *documents* do not. Getting this backwards measurably hurts retrieval, so it
# lives here and both sides import it.
QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _text(value) -> str:
    """Coerce any CSV cell to a string. Handles None, NaN and pandas.NA.

    pandas 3.x reads missing strings as `pd.NA`, whose truthiness raises
    "boolean value of NA is ambiguous", so the usual `value or ""` is unsafe.
    """
    if value is None:
        return ""
    try:
        if value is not value:            # NaN
            return ""
    except Exception:                      # pragma: no cover - exotic scalars
        pass
    if str(type(value)).endswith("NAType'"):
        return ""
    return str(value).strip()


def format_query(question: str, use_instruction: bool = True) -> str:
    """Embed-ready text for a user question."""
    question = _text(question)
    if not question:
        return ""
    if not use_instruction:
        return question
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"


def document_text(title, synopsis, topic: str = "") -> str:
    """The text we index for one article. Title first - it carries most of the signal."""
    title, synopsis, topic = _text(title), _text(synopsis), _text(topic)
    return f"{title}. {synopsis}".strip() or title or topic


# ---------------------------------------------------------------------------
# MySQL (optional - Phase 1 runs on the CSV when no server is reachable)
# ---------------------------------------------------------------------------
MYSQL = {
    "host": os.environ.get("ET_MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("ET_MYSQL_PORT", "3306")),
    "user": os.environ.get("ET_MYSQL_USER", "root"),
    "password": os.environ.get("ET_MYSQL_PASSWORD", ""),
    "database": os.environ.get("ET_MYSQL_DB", "et_news"),
    "connect_timeout": int(os.environ.get("ET_MYSQL_TIMEOUT", "3")),
}

# ---------------------------------------------------------------------------
# CSV schema (the documented Data/articles_clean.csv columns)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "Topic",
    "Title",
    "Article_URL",
    "Publication_Date",
    "Extraction_Date",
    "Synopsis",
]

CSV_DTYPES = {
    "Topic": "string",
    "Title": "string",
    "Article_URL": "string",
    "Publication_Date": "string",
    "Extraction_Date": "string",
    "Synopsis": "string",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
# Every ET date in the corpus has the same shape:  "20 Jul, 2026, 04.16 PM IST"
# The trailing timezone token is what makes pandas (and dateutil) fail on it, so
# it is stripped before parsing and the value is kept as a naive publisher
# wall-clock time - that is what the UI shows and what the date filters compare.
TZ_SUFFIX = re.compile(r"\s*(IST|UTC|GMT)\s*$", re.I)

PUB_DATE_RE = re.compile(
    r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s*,?\s*(\d{4})"          # day, month, year
    r"(?:\s*,?\s*(\d{1,2})[.:](\d{2})(?:[.:](\d{2}))?\s*([AaPp][Mm])?)?",  # optional time
)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_publication_date(raw) -> datetime | None:
    """
    '07 Oct, 2025, 09.49 AM IST' -> datetime(2025, 10, 7, 9, 49), or None.

    The single date implementation for the whole project. `clean_data.py`,
    `check_data.py`, `Database_Connection.py` and the Streamlit app all import
    it from here so a format change is a one-line fix.
    """
    if raw is None or not isinstance(raw, str):
        return None
    text = TZ_SUFFIX.sub("", raw).strip()
    m = PUB_DATE_RE.match(text)
    if not m:
        return None

    month = _MONTHS.get(m.group(2).lower())
    if month is None:
        return None

    hour, minute, second = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
    meridiem = (m.group(7) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    try:
        return datetime(int(m.group(3)), month, int(m.group(1)), hour, minute, second)
    except ValueError:
        return None


def parse_publication_series(series):
    """
    Vectorised wrapper: a pandas Series of date strings -> datetime Series.

    Falls back to the scalar parser for anything pandas itself cannot handle,
    so the result is never silently all-NaT.
    """
    import pandas as pd

    stripped = series.astype("string").str.replace(TZ_SUFFIX, "", regex=True)
    parsed = pd.to_datetime(stripped, format="mixed", errors="coerce")

    missing = parsed.isna() & stripped.notna()
    if missing.any():
        parsed = parsed.copy()
        for idx in series.index[missing]:
            value = parse_publication_date(series.at[idx])
            if value is not None:
                parsed.at[idx] = pd.Timestamp(value)
    return parsed