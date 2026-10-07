"""
Database_Connection.py
===========================================================================
MySQL access for ET News, using `mysql-connector-python`.

Design rule: **this module never raises on a missing server.** The Phase 1
project is documented as running off `Data/articles_clean.csv`, and there is no
MySQL service on the development machine, so every entry point degrades to CSV
with an informational message instead of a stack trace.

Usage
-----
    from Database_Connection import get_connection, load_articles

    conn = get_connection()          # -> connection, or None
    df   = load_articles()           # -> DataFrame from MySQL if up, else the CSV

Enable it by setting the environment (or editing config.MYSQL):

    $env:ET_MYSQL_USER = "root"
    $env:ET_MYSQL_PASSWORD = "secret"
    $env:ET_MYSQL_DB = "et_news"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    Topic             VARCHAR(120)  NOT NULL,
    Title             VARCHAR(512)  NOT NULL,
    Article_URL       VARCHAR(768)  NOT NULL UNIQUE,
    Publication_Date  VARCHAR(64)   NULL,
    Extraction_Date   VARCHAR(64)   NULL,
    Synopsis          TEXT          NULL,
    publication_iso   DATETIME      NULL,
    extraction_iso    DATETIME      NULL,
    embedded          TINYINT(1)    NOT NULL DEFAULT 0,
    INDEX idx_news_publication (publication_iso),
    INDEX idx_news_topic (Topic),
    INDEX idx_news_embedded (embedded)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _driver():
    """Import mysql.connector lazily so the CSV path never needs it installed."""
    try:
        import mysql.connector
        from mysql.connector import Error
        return mysql.connector, Error
    except ImportError:
        return None, None


def get_connection(database: str | None = None):
    """
    Return a live MySQL connection, or None.

    Every failure mode - driver missing, server down, bad credentials, unknown
    database - collapses to a single warning so callers only need one check.
    """
    mysql, Error = _driver()
    if mysql is None:
        print("[db] mysql-connector-python not installed - falling back to the CSV dataset")
        return None

    cfg = dict(config.MYSQL)
    if database:
        cfg["database"] = database

    try:
        conn = mysql.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            connection_timeout=cfg["connect_timeout"],
        )
    except Error as exc:
        print(f"[db] no MySQL server at {cfg['host']}:{cfg['port']} ({exc}) - using the CSV dataset")
        return None

    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cur.close()
        conn.database = cfg["database"]
        cur = conn.cursor()
        for statement in filter(None, (s.strip() for s in SCHEMA_SQL.split(";"))):
            cur.execute(statement)
        conn.commit()
        cur.close()
        print(f"[db] connected to MySQL {cfg['host']}:{cfg['port']}/{cfg['database']}")
        return conn
    except Error as exc:
        print(f"[db] schema setup failed ({exc}) - using the CSV dataset")
        try:
            conn.close()
        except Error:
            pass
        return None


def is_available() -> bool:
    """True if a MySQL server answers right now (closes the probe connection)."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        conn.close()
    except Exception:
        pass
    return True


def load_articles() -> pd.DataFrame:
    """
    The article corpus, from MySQL when a server is reachable, otherwise from
    `Data/articles_clean.csv`.

    Both paths return the same six documented columns plus two parsed datetime
    columns used for filtering, so callers never branch.
    """
    conn = get_connection()
    if conn is not None:
        try:
            df = pd.read_sql("SELECT Topic, Title, Article_URL, Publication_Date, "
                             "Extraction_Date, Synopsis FROM news", conn)
            print(f"[db] loaded {len(df):,} articles from MySQL")
            return _finalise(df)
        except Exception as exc:                       # noqa: BLE001
            print(f"[db] query failed ({exc}) - using the CSV dataset")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return load_articles_csv()


def load_articles_csv(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.ARTICLES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"[data] {path} not found. Run:\n"
            f"       python src/combine_data.py\n"
            f"       python src/clean_data.py"
        )
    df = pd.read_csv(path, dtype=config.CSV_DTYPES, keep_default_na=False, na_values=[""])
    print(f"[data] loaded {len(df):,} articles from {path.name}")
    return _finalise(df)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the two date columns into real datetimes for filtering."""
    df = df.copy()
    df["publication_dt"] = config.parse_publication_series(df["Publication_Date"])
    df["extraction_dt"] = config.parse_publication_series(df["Extraction_Date"])
    return df


def upsert_articles(df: pd.DataFrame, batch_size: int = 500) -> int:
    """
    Insert any article whose URL is not already stored. Returns rows added.

    Used by the RSS collector once a MySQL server exists; Phase 1 does not
    call it.
    """
    conn = get_connection()
    if conn is None:
        return 0

    _, Error = _driver()
    added = 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT Article_URL FROM news")
        known = {row[0] for row in cur.fetchall()}

        pending = [r for r in df.to_dict("records") if r.get("Article_URL") not in known]
        sql = (
            "INSERT IGNORE INTO news "
            "(Topic, Title, Article_URL, Publication_Date, Extraction_Date, Synopsis, publication_iso) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            rows = [
                (
                    r.get("Topic"), r.get("Title"), r.get("Article_URL"),
                    r.get("Publication_Date"), r.get("Extraction_Date"),
                    r.get("Synopsis"),
                    None if r.get("publication_dt") is None or pd.isna(r.get("publication_dt"))
                    else pd.Timestamp(r["publication_dt"]).to_pydatetime(),
                )
                for r in chunk
            ]
            cur.executemany(sql, rows)
            conn.commit()
            added += cur.rowcount

        cur.close()
        print(f"[db] inserted {added:,} new articles ({len(df) - added:,} already present)")
        return added
    except Error as exc:
        print(f"[db] upsert failed: {exc}")
        conn.rollback()
        return added
    finally:
        try:
            conn.close()
        except Exception:
            pass


def articles_since(df: pd.DataFrame, last_run) -> pd.DataFrame:
    """Rows newer than `last_run` - what the daily job actually wants to file."""
    if last_run is None:
        return df
    return df[df["publication_dt"] > pd.Timestamp(last_run)]


if __name__ == "__main__":
    print("=" * 70)
    print("MySQL status")
    print("=" * 70)
    print(f"  target     : {config.MYSQL['host']}:{config.MYSQL['port']}/{config.MYSQL['database']}")
    print(f"  driver     : {'installed' if _driver()[0] else 'NOT installed'}")
    print(f"  server     : {'reachable' if is_available() else 'not reachable'}")
    print(f"  app will use: {'MySQL' if is_available() else 'Data/articles_clean.csv'}")
    print("=" * 70)