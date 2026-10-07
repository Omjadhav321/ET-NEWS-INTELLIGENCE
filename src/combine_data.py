"""
combine_data.py
===========================================================================
Step 1 of the data pipeline.

Reads every `*.json` scrape in `ET_Data_fetch.json/` (21 files, 26,484 rows)
and stacks them into ONE verbatim table at `Data/articles_raw.csv`.

Deliberately *no* cleaning here. This stage answers exactly one question - how
much raw material is there, and does it parse - so any later data quality
problem can be attributed to `clean_data.py` rather than to the reader.

It also accepts the consolidated `data/articles.json` (produced by the Node
`build_data.js`) via --from-json, which skips re-parsing 21 files.

Usage
-----
    python src/combine_data.py
    python src/combine_data.py --from-json
    python src/combine_data.py --out Data/articles_raw.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config


def topic_from_filename(path: Path) -> str:
    """`Cyber_Security_20260720_074319.json` -> `Cyber Security`."""
    return path.stem.split("_20")[0].replace("_", " ").strip()


def load_raw_json_folder(folder: Path) -> list[dict]:
    """Every parseable record from every .json file in `folder`, verbatim."""
    if not folder.is_dir():
        raise SystemExit(f"[combine] Raw folder not found: {folder}")

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() == ".json")
    if not files:
        raise SystemExit(f"[combine] No .json files inside {folder}")

    rows: list[dict] = []
    for path in files:
        # utf-8-sig transparently eats the BOM that some scrapes emit.
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            print(f"[combine]   skip {path.name}: invalid JSON ({exc})")
            continue

        records = payload if isinstance(payload, list) else [payload]
        fallback_topic = topic_from_filename(path)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            # Some scrapes prefix the key itself: {"\ufeffTopic": ...}
            rec = {(k.lstrip("\ufeff") if isinstance(k, str) else k): v
                   for k, v in rec.items()}
            rec.setdefault("Topic", fallback_topic)
            rec["_source_file"] = path.name
            rows.append(rec)
        print(f"[combine]   {path.name:<48} {len(records):>6} rows")

    return rows


def load_from_consolidated(json_path: Path) -> list[dict]:
    """Map the Node build's compact schema back onto the documented one."""
    articles = json.loads(json_path.read_text(encoding="utf-8"))
    rows = [
        {
            "Topic": a.get("topic", ""),
            "Title": a.get("title", ""),
            "Article_URL": a.get("url", ""),
            "Publication_Date": a.get("date", ""),
            "Extraction_Date": "",                 # not carried by the JSON build
            "Synopsis": a.get("synopsis", ""),
            "_source_file": "articles.json",
        }
        for a in articles
    ]
    print(f"[combine]   articles.json -> {len(rows)} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Combine the raw ET scrapes into one table")
    ap.add_argument("--src", default=str(config.RAW_DIR), help="folder of raw *.json scrapes")
    ap.add_argument("--out", default=str(config.ARTICLES_RAW_CSV), help="output CSV")
    ap.add_argument("--from-json", action="store_true",
                    help="read the consolidated data/articles.json instead of the raw folder")
    args = ap.parse_args()

    config.ensure_dirs()
    started = time.time()

    if args.from_json:
        if not config.RAW_JSON.exists():
            raise SystemExit(f"[combine] {config.RAW_JSON} not found")
        rows = load_from_consolidated(config.RAW_JSON)
    else:
        rows = load_raw_json_folder(Path(args.src))

    if not rows:
        raise SystemExit("[combine] Nothing was parsed - aborting")

    # Documented columns first, then any unexpected extra keys, so a schema
    # surprise shows up in the report rather than being silently dropped.
    extras = sorted({k for r in rows for k in r} - set(config.CSV_COLUMNS) - {"_source_file"})
    df = pd.DataFrame(rows)
    df = df[[c for c in config.CSV_COLUMNS if c in df.columns] + extras]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    mb = out.stat().st_size / 1048576
    print("\n[combine] ------------------------------------------------------")
    print(f"[combine] files read      : {len({r['_source_file'] for r in rows})}")
    print(f"[combine] raw rows        : {len(df):,}")
    print(f"[combine] unexpected cols : {extras or '(none)'}")
    print("[combine] missing values  :")
    for col in config.CSV_COLUMNS:
        if col in df.columns:
            n = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
            print(f"[combine]     {col:<20} {n:>7,}")
    print(f"[combine] output          : {out.name} ({mb:.2f} MB, {time.time() - started:.1f}s)")
    print("[combine] Next: python src/clean_data.py")
    print("[combine] ------------------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())