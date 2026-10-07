"""
check_data.py
===========================================================================
Validation suite to ensure the cleaned dataset is safe for embeddings.
"""
from __future__ import annotations

import pandas as pd
import config

def verify():
    print("[verify] auditing cleaned dataset...")
    
    if not config.ARTICLES_CSV.exists():
        print(f"[verify] error: {config.ARTICLES_CSV} not found.")
        return

    df = pd.read_csv(config.ARTICLES_CSV, dtype=config.CSV_DTYPES, keep_default_na=False, na_values=[""])
    
    # Check 1: Column presence
    missing_cols = [c for c in config.CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[verify] FAILED: missing columns {missing_cols}")
        return

    # Check 2: No nulls in required fields
    null_titles = df["Title"].isna().sum()
    null_urls = df["Article_URL"].isna().sum()
    if null_titles > 0 or null_urls > 0:
        print(f"[verify] FAILED: found nulls in Title({null_titles}) or URL({null_urls})")
        return

    # Check 3: No duplicate URLs
    if df["Article_URL"].duplicated().any():
        print("[verify] FAILED: duplicate URLs found")
        return

    print(f"[verify] PASSED: {len(df):,} articles validated.")

if __name__ == "__main__":
    verify()
