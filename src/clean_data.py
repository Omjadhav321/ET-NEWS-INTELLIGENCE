"""
clean_data.py
===========================================================================
Performs deduplication, missing value handling, and date normalization.
Reads from Data/articles_raw.csv -> Writes to Data/articles_clean.csv.
"""
from __future__ import annotations

import pandas as pd
import config

def clean():
    print("[clean] starting data cleaning pipeline...")
    
    if not config.ARTICLES_RAW_CSV.exists():
        print(f"[clean] error: {config.ARTICLES_RAW_CSV} not found. Run combine_data.py first.")
        return

    # 1. Load raw data
    df = pd.read_csv(config.ARTICLES_RAW_CSV, dtype=config.CSV_DTYPES, keep_default_na=False, na_values=[""])
    initial_count = len(df)
    print(f"[clean] loaded {initial_count:,} rows from raw data")

    # 2. Remove exact duplicates
    df = df.drop_duplicates()
    
    # 3. Remove duplicate Article URLs (keep first)
    df = df.drop_duplicates(subset=["Article_URL"], keep="first")
    
    # 4. Handle missing values for critical columns
    # We require Title and Article_URL. Others can be empty.
    df = df[df["Title"].notna() & (df["Title"] != "")]
    df = df[df["Article_URL"].notna() & (df["Article_URL"] != "")]
    
    # 5. Normalize text (strip whitespace)
    for col in ["Topic", "Title", "Synopsis"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    final_count = len(df)
    removed = initial_count - final_count
    
    # Save cleaned dataset
    df.to_csv(config.ARTICLES_CSV, index=False)
    print(f"[clean] removed {removed:,} duplicates/invalid rows.")
    print(f"[clean] successfully wrote {final_count:,} articles to {config.ARTICLES_CSV}")

if __name__ == "__main__":
    config.ensure_dirs()
    clean()
