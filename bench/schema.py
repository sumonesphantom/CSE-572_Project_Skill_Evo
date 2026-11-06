"""Schema renderings for prompt-engineering ablations.

- ddl            : raw CREATE TABLE statements only.
- ddl_samples    : the repo default (DDL + 2 sample rows per table, skillevo.sqlexec.schema_text, 6,000-char cap).
- ddl_samples_full: the same with the 20,000-char cap mschema uses (control for the cap).
- mschema        : compact per-column listing with type, PK, distinct example values and FK lines
                   (an M-Schema-style rendering, as used by XiYan-SQL / CodeS-style prompts).
- mschema_desc   : mschema + BIRD column descriptions / value descriptions from
                   database_description/*.csv (Spider has none, so it falls back to mschema).
"""
from __future__ import annotations

import csv
import sqlite3
from functools import lru_cache
from pathlib import Path

from skillevo.sqlexec import schema_text


def _q(name: str) -> str:
    return f'"{name}"' if not name.replace("_", "").isalnum() else name


def _short(v, n: int = 30) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


@lru_cache(maxsize=64)
def _descriptions(db_path: str) -> dict[tuple[str, str], str]:
    d = Path(db_path).parent / "database_description"
    out: dict[tuple[str, str], str] = {}
    if not d.is_dir():
        return out
    for f in d.glob("*.csv"):
        table = f.stem.lower()
        for enc in ("utf-8-sig", "latin-1"):
            try:
                with open(f, newline="", encoding=enc) as fh:
                    rows = list(csv.DictReader(fh))
                break
            except UnicodeDecodeError:
                continue
        else:
            continue
        for r in rows:
            col = (r.get("original_column_name") or "").strip()
            if not col:
                continue
            parts = []
            full = (r.get("column_name") or "").strip()
            desc = (r.get("column_description") or "").strip()
            vdesc = (r.get("value_description") or "").strip()
            if full and full.lower() != col.lower():
                parts.append(full)
            if desc and desc.lower() not in (col.lower(), full.lower()):
                parts.append(desc)
            if vdesc:
                parts.append("values: " + vdesc)
            if parts:
                out[(table, col.lower())] = _short(" | ".join(parts), 160)
    return out


@lru_cache(maxsize=256)
def mschema(db_path: str, n_examples: int = 3, with_desc: bool = False, max_chars: int = 20000) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    descs = _descriptions(db_path) if with_desc else {}
    lines, fks = [], []
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        for t in tables:
            lines.append(f"# Table: {_q(t)}")
            for cid, col, typ, notnull, _dflt, pk in conn.execute(f'PRAGMA table_info("{t}")'):
                bits = [typ or "ANY"]
                if pk:
                    bits.append("PK")
                try:
                    vals = [r[0] for r in conn.execute(
                        f'SELECT DISTINCT "{col}" FROM "{t}" WHERE "{col}" IS NOT NULL LIMIT {n_examples}')]
                except sqlite3.Error:
                    vals = []
                if vals:
                    bits.append("e.g. " + ", ".join(repr(_short(v)) if isinstance(v, str) else _short(v) for v in vals))
                d = descs.get((t.lower(), col.lower()))
                line = f"  {_q(col)}: " + "; ".join(bits)
                lines.append(line + (f"  -- {d}" if d else ""))
            for row in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
                _id, _seq, ref_t, frm, to = row[:5]
                fks.append(f"{_q(t)}.{_q(frm)} = {_q(ref_t)}.{_q(to) if to else '?'}")
    finally:
        conn.close()
    if fks:
        lines.append("# Foreign keys:")
        lines.extend("  " + f for f in fks)
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + "\n/* schema truncated */"


def render(db_path: str, style: str) -> str:
    if style == "ddl":
        return schema_text(db_path, 0)
    if style == "ddl_samples":
        return schema_text(db_path, 2)
    if style == "ddl_samples_full":
        return schema_text(db_path, 2, max_chars=20000)
    if style == "mschema":
        return mschema(db_path)
    if style == "mschema_desc":
        return mschema(db_path, with_desc=True)
    raise ValueError(f"unknown schema style {style!r}")
