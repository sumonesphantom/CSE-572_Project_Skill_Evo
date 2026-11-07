"""Schema renderings for the prompt.

- repo    : the repo default (skillevo.sqlexec.schema_text: DDL + 2 sample rows, 6,000-char cap).
            Byte-identical to the SkillEvo solver, so A1 is the same prompt Venkat's P1 used.
- ddl     : CREATE TABLE statements only.
- mschema : one line per column with type, primary key and up to 3 distinct example values,
            then the foreign-key edges. Compact and explicit about which column sits in which
            table, and how tables join.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from skillevo.sqlexec import schema_text

MAX_CHARS = 24_000


def quote_ident(name: str) -> str:
    return name if name.replace("_", "").isalnum() and not name[0].isdigit() else f"`{name}`"


def _val(v, n: int = 40) -> str:
    if isinstance(v, str):
        s = v.replace("\n", " ")
        s = s if len(s) <= n else s[:n] + "…"
        return "'" + s.replace("'", "''") + "'"
    return str(v)


def _connect(db_path: str):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


@lru_cache(maxsize=128)
def _tables(db_path: str) -> tuple[tuple[str, str], ...]:
    conn = _connect(db_path)
    try:
        return tuple(conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                                  "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall())
    finally:
        conn.close()


@lru_cache(maxsize=64)
def mschema(db_path: str, n_examples: int = 3, max_chars: int = MAX_CHARS) -> str:
    conn = _connect(db_path)
    out, fks = [], []
    try:
        for table, _ in _tables(db_path):
            qt = quote_ident(table)
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()  # cid,name,type,notnull,dflt,pk
            pk_cols = [c[1] for c in cols if c[5]]
            lines = [f"Table {qt} ("]
            for _, name, ctype, _nn, _d, pk in cols:
                desc = f"  {quote_ident(name)} {ctype or 'ANY'}"
                if pk:
                    desc += " PRIMARY KEY" if len(pk_cols) == 1 else " PRIMARY KEY (composite)"
                if n_examples:
                    try:
                        vals = conn.execute(f'SELECT DISTINCT "{name}" FROM "{table}" WHERE "{name}" IS NOT NULL '
                                            f"AND \"{name}\" != '' LIMIT {n_examples}").fetchall()
                        if vals:
                            desc += "  -- e.g. " + ", ".join(_val(v[0]) for v in vals)
                    except sqlite3.Error:
                        pass
                lines.append(desc)
            lines.append(")")
            out.append("\n".join(lines))
            for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
                # id, seq, ref_table, from_col, to_col, ...; to_col is None when it refers to the parent PK
                ref, src, dst = fk[2], fk[3], fk[4]
                if dst is None:
                    ref_pk = [c[1] for c in conn.execute(f'PRAGMA table_info("{ref}")').fetchall() if c[5]]
                    dst = ref_pk[0] if ref_pk else "?"
                fks.append(f"{qt}.{quote_ident(src)} = {quote_ident(ref)}.{quote_ident(dst)}")
    finally:
        conn.close()
    text = "\n\n".join(out)
    if fks:
        text += "\n\nForeign keys:\n" + "\n".join(dict.fromkeys(fks))
    return text if len(text) <= max_chars else text[:max_chars] + "\n/* schema truncated */"


def ddl(db_path: str) -> str:
    return "\n".join(s.strip() + ";" for _, s in _tables(db_path) if s)


def render(db_path: str, kind: str) -> str:
    if kind == "repo":
        return schema_text(db_path, 2)
    if kind == "ddl":
        return ddl(db_path)
    if kind == "mschema":
        return mschema(db_path)
    raise ValueError(f"unknown schema kind {kind!r}")
