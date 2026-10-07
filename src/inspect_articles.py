"""
inspect_articles.py
===========================================================================
Utility to peek at the dataset for manual verification.
"""
from __future__ import annotations

import pandas as pd
import config

def peek(n=5):
    if not config.ARTICLES_CSV.exists():
        print(f"Error: {config.ARTICLES_CSV} not found.")
        return
    
    df = pd.read_csv(config.ARTICLES_CSV, dtype=config.CSV_DTYPES, keep_default_na=False, na_values=[""])
    print(f"--- Dataset Preview (Top {n}) ---")
    print(df.head(n).to_string())
    print("-" * 30)
    print(f"Total Articles: {len(df):,}")
    print(f"Unique Topics: {df['Topic'].nunique()}")

if __name__ == "__main__":
    peek()
