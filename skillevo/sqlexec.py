"""SQLite execution and execution-accuracy (EX) scoring, BIRD-style."""
from __future__ import annotations

import sqlite3
import time
from functools import lru_cache


class SQLTimeout(Exception):
    pass


def execute(db_path: str, sql: str, timeout_s: float = 30.0, max_rows: int = 100_000):
    """Run a read-only query. Returns (rows, error_message_or_None)."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return None, f"cannot open db: {e}"
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return rows, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg:
            msg = f"query timed out after {timeout_s:.0f}s"
        return None, msg
    except Exception as e:  # noqa: BLE001 - any driver error is feedback for the agent
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


@lru_cache(maxsize=4096)
def gold_result(db_path: str, gold_sql: str):
    rows, err = execute(db_path, gold_sql, timeout_s=120)
    if err:
        return None
    return frozenset(rows)


def is_correct(db_path: str, pred_sql: str, gold_sql: str, timeout_s: float = 30.0):
    """Returns (correct: bool, error: str|None, n_rows: int)."""
    rows, err = execute(db_path, pred_sql, timeout_s)
    if err:
        return False, err, 0
    gold = gold_result(db_path, gold_sql)
    if gold is None:
        return False, "gold query failed", len(rows)
    return frozenset(rows) == gold, None, len(rows)


def schema_text(db_path: str, sample_rows: int = 2, max_chars: int = 6000) -> str:
    return _schema_text(db_path, sample_rows, max_chars)


@lru_cache(maxsize=256)
def _schema_text(db_path: str, sample_rows: int, max_chars: int) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    parts = []
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for name, ddl in tables:
            parts.append(ddl.strip() + ";")
            if sample_rows:
                try:
                    rows = conn.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}').fetchall()
                    if rows:
                        shown = [tuple(str(v)[:40] for v in r) for r in rows]
                        parts.append(f"/* sample rows from {name}: {shown} */")
                except sqlite3.Error:
                    pass
    finally:
        conn.close()
    text = "\n".join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + "\n/* schema truncated */"
