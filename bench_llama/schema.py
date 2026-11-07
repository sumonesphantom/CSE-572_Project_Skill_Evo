"""Compact, optionally pruned schema for small models.

The repo serialises full CREATE TABLE statements + 2 sample rows and cuts at 6000 chars, which
silently drops whole tables on 5 BIRD databases. Here every table is kept in a compact form, and
`prune` keeps only the tables most related to the question (lexical schema linking).
"""
from __future__ import annotations

import csv
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

STOP = set("a an the of in on for to and or is are was were be by with what which who how many much does do did "
           "from that this as at it its list give show find name all each their there than please".split())


def toks(text: str) -> set:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)              # camelCase -> camel Case
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")):
        if w in STOP or len(w) < 2:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def q(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f"`{name}`"


def _read_desc(db_path: str) -> dict:
    """BIRD ships database_description/<table>.csv with human column descriptions."""
    d = Path(db_path).parent / "database_description"
    out = {}
    if not d.exists():
        return out
    for f in d.glob("*.csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                rows = list(csv.DictReader(open(f, encoding=enc)))
                break
            except UnicodeDecodeError:
                continue
        for r in rows:
            col = (r.get("original_column_name") or "").strip()
            if not col:
                continue
            desc = " ".join(x.strip() for x in (r.get("column_description"), r.get("value_description")) if x and x.strip())
            out[(f.stem.lower(), col.lower())] = re.sub(r"\s+", " ", desc)
    return out


@lru_cache(maxsize=64)
def db_info(db_path: str, n_values: int = 2):
    """[(table, [(col, type, pk, fk_target|None, examples, description)])]"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    desc = _read_desc(db_path)
    tables = []
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in names:
            fks = {r[3]: f"{r[2]}({r[4]})" for r in conn.execute(f'PRAGMA foreign_key_list("{t}")') if r[4]}
            cols = []
            for _, c, typ, _, _, pk in conn.execute(f'PRAGMA table_info("{t}")'):
                ex = []
                if n_values and (typ or "").upper() in ("TEXT", "") and not pk:
                    try:
                        ex = [str(v[0])[:30] for v in conn.execute(
                            f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL AND "{c}" != "" LIMIT {n_values}')]
                    except sqlite3.Error:
                        pass
                cols.append((c, typ or "", bool(pk), fks.get(c), ex, desc.get((t.lower(), c.lower()), "")))
            tables.append((t, cols))
    finally:
        conn.close()
    return tables


def _useful(desc: str, col: str) -> str:
    desc = re.sub(r"commonsense (reasoning|evidence):?", "", desc, flags=re.I).strip(" ;:")
    if not desc or toks(desc) <= toks(col) | {"id", "unique", "identification", "number"}:
        return ""
    return desc[:60]


def render(tables, descriptions: bool = True, values: bool = True) -> str:
    out = []
    for t, cols in tables:
        lines = []
        for c, typ, pk, fk, ex, desc in cols:
            s = f"{q(c)} {typ}".rstrip()
            if pk:
                s += " PRIMARY KEY"
            if fk:
                s += f" REFERENCES {fk}"
            notes = []
            d = _useful(desc, c) if descriptions else ""
            if d:
                notes.append(d)
            if values and ex:
                notes.append("e.g. " + ", ".join(repr(e) for e in ex))
            lines.append((s, "; ".join(notes)))
        body = []
        for i, (s, note) in enumerate(lines):
            comma = "," if i < len(lines) - 1 else ""
            body.append(f"  {s}{comma}" + (f"  -- {note}" if note else ""))
        out.append(f"CREATE TABLE {q(t)} (\n" + "\n".join(body) + "\n);")
    return "\n".join(out)


def link(tables, question: str, max_tables: int):
    """Keep the max_tables tables most related to the question, plus their foreign-key targets.
    Score = distinct question words covered by the table (name, columns, descriptions),
    table-name matches x3, and +5 for every column/table named verbatim in the question/hint."""
    qt = toks(question)
    ql = question.lower()
    scored = []
    for t, cols in tables:
        covered = set()
        verbatim = 0
        for c, _, _, _, _, d in cols:
            covered |= qt & (toks(c) | toks(d))
            if len(c) > 3 and re.search(r"(?<![a-z0-9_])" + re.escape(c.lower()) + r"(?![a-z0-9_])", ql):
                verbatim += 1
        if re.search(r"(?<![a-z0-9_])" + re.escape(t.lower()) + r"(?![a-z0-9_])", ql):
            verbatim += 1
        scored.append((3 * len(qt & toks(t)) + len(covered) + 5 * verbatim, t))
    scored.sort(key=lambda x: -x[0])
    keep = {t for _, t in scored[:max_tables]}
    for t, cols in tables:
        if t in keep:
            keep |= {c[3].split("(")[0] for c in cols if c[3]}
    return [(t, c) for t, c in tables if t in keep]


def schema_for(task, prune: int = 0, descriptions: bool = True, values: bool = True) -> str:
    tables = db_info(task.db_path)
    if prune and len(tables) > prune:
        tables = link(tables, f"{task.question} {task.evidence}", prune)
    return render(tables, descriptions, values)
