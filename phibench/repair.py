"""B1: schema-aware repair feedback.

The plain repair turn passes on SQLite's message ("no such column: T1.FundingType"), and in
Phase A Phi-4-mini turned that into a correct answer only 2 times in 78 on BIRD test. Most of
those errors name a column that exists, but in another table than the alias points to. Here the
harness looks the problem up in the database and says exactly what is wrong: which table the
alias refers to, which table(s) actually hold the column, close name matches, and SQLite
equivalents for functions it does not have. All deterministic: no extra model calls.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
from functools import lru_cache

from .schema import quote_ident

_STOP = r"(?:ON|WHERE|JOIN|INNER|LEFT|RIGHT|CROSS|FULL|OUTER|NATURAL|GROUP|ORDER|LIMIT|UNION|INTERSECT|EXCEPT|HAVING|USING|AND|OR)"
_FROM = re.compile(r"\b(?:FROM|JOIN)\s+(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|\w+)"
                   rf"(?:\s+(?:AS\s+)?(?!{_STOP}\b)(\w+))?", re.I)
FUNCS = {
    "year": "strftime('%Y', col)", "month": "strftime('%m', col)", "day": "strftime('%d', col)",
    "concat": "a || b", "datediff": "julianday(a) - julianday(b)", "now": "date('now')",
    "isnull": "IFNULL(a, b)", "if": "CASE WHEN ... THEN ... ELSE ... END", "len": "LENGTH(x)",
    "left": "SUBSTR(x, 1, n)", "right": "SUBSTR(x, -n)", "to_date": "date(x)", "getdate": "date('now')",
    "datepart": "strftime(...)", "extract": "strftime(...)", "top": "ORDER BY ... LIMIT n",
}


@lru_cache(maxsize=128)
def table_columns(db_path: str) -> dict[str, list[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')] for t in tables}
    finally:
        conn.close()


def _unq(name: str) -> str:
    return name.strip().strip("`\"[]")


def aliases(sql: str, tables: dict[str, list[str]]) -> dict[str, tuple[str, str]]:
    """alias (lower) -> (real table name, alias as written), for tables named in FROM / JOIN."""
    lower = {t.lower(): t for t in tables}
    out = {}
    for tbl, alias in _FROM.findall(sql):
        real = lower.get(_unq(tbl).lower())
        if not real:
            continue
        out.setdefault(real.lower(), (real, real))
        if alias:
            out[alias.lower()] = (real, alias)
    return out


def _cols_line(table: str, cols: list[str], limit: int = 40) -> str:
    shown = ", ".join(quote_ident(c) for c in cols[:limit]) + (" ..." if len(cols) > limit else "")
    return f"Columns of {quote_ident(table)}: {shown}"


def _close(name: str, tables: dict[str, list[str]], n: int = 3) -> list[str]:
    pool = {f"{quote_ident(t)}.{quote_ident(c)}": c for t, cs in tables.items() for c in cs}
    low = name.lower()
    starts = [k for k, c in pool.items() if c.lower().startswith(low) and c.lower() != low]
    fuzzy = difflib.get_close_matches(low, [c.lower() for c in pool.values()], n=n, cutoff=0.75)
    fuzzy_keys = [k for k, c in pool.items() if c.lower() in fuzzy]
    return list(dict.fromkeys(starts + fuzzy_keys))[:n]


def unquoted_special(sql: str, tables: dict[str, list[str]]) -> list[str]:
    """Names with spaces/symbols that appear in the SQL without backticks/quotes/brackets."""
    found = []
    for cols in tables.values():
        for c in cols:
            if quote_ident(c) == c:
                continue
            for m in re.finditer(re.escape(c), sql):
                before = sql[m.start() - 1] if m.start() else ""
                if before not in "`\"[":
                    found.append(c)
                    break
    return list(dict.fromkeys(found))


def diagnose(db_path: str, sql: str, err: str) -> str:
    """Extra, database-grounded hints for an SQLite error message ('' if none apply)."""
    tables = table_columns(db_path)
    amap = aliases(sql, tables)
    e = err.lower()
    hints: list[str] = []
    m = re.search(r"no such column: ([^\s]+)", err, re.I)
    if m:
        ref = m.group(1)
        alias, col = (ref.split(".", 1) + [""])[:2] if "." in ref else ("", ref)
        col = _unq(col)
        holders = [t for t, cs in tables.items() if col.lower() in (c.lower() for c in cs)]
        if alias and alias.lower() in amap:
            t = amap[alias.lower()][0]
            hints.append(f"In your query, {alias} refers to table {quote_ident(t)}, which has no column {quote_ident(col)}.")
            hints.append(_cols_line(t, tables[t]))
        elif alias:
            hints.append(f"{alias} is not a table or alias defined in your FROM/JOIN clauses.")
        if holders:
            where = ", ".join(quote_ident(t) for t in holders)
            used = [w for a, (t, w) in amap.items() if t in holders and a != t.lower()]
            tip = f" (in your query: {', '.join(used)})" if used else " - join that table if you need it"
            hints.append(f"Column {quote_ident(col)} exists in: {where}{tip}.")
        else:
            near = _close(col, tables)
            if near:
                hints.append(f"There is no column named {quote_ident(col)}. Closest names: {', '.join(near)}.")
            else:
                hints.append(f"There is no column named {quote_ident(col)} in any table.")
    m = re.search(r"no such table: ([^\s]+)", err, re.I)
    if m:
        name = _unq(m.group(1).split(".")[-1])
        near = difflib.get_close_matches(name.lower(), [t.lower() for t in tables], n=3, cutoff=0.6)
        hints.append(f"Tables in this database: {', '.join(quote_ident(t) for t in tables)}."
                     + (f" Closest to {name}: {', '.join(near)}." if near else ""))
    m = re.search(r"ambiguous column name: ([^\s]+)", err, re.I)
    if m:
        col = _unq(m.group(1))
        holders = [w for a, (t, w) in amap.items() if col.lower() in (c.lower() for c in tables[t]) and a != t.lower()]
        hints.append(f"Several tables in your query have {quote_ident(col)}; prefix it with the right alias"
                     + (f" ({', '.join(holders)})." if holders else "."))
    m = re.search(r"no such function: (\w+)", err, re.I)
    if m:
        f = m.group(1).lower()
        hints.append(f"SQLite has no {m.group(1).upper()}() function." + (f" Use {FUNCS[f]} instead." if f in FUNCS else ""))
    if "syntax error" in e or "unrecognized token" in e or "incomplete input" in e:
        bad = unquoted_special(sql, tables)
        if bad:
            hints.append("These names contain spaces or symbols and must be wrapped in backticks: "
                         + ", ".join(f"`{c}`" for c in bad) + ".")
        hints.append("Check that every bracket is closed and that only SQLite syntax is used.")
    return "\n".join(hints)


def error_message(db_path: str, sql: str, err: str) -> str:
    extra = diagnose(db_path, sql, err)
    return (f"That query failed with: {err}\n" + (extra + "\n" if extra else "")
            + "Fix it. Return only the corrected ```sql block.")


EMPTY_MESSAGE = ("That query ran but returned no rows. The question expects an answer, so a filter, join or "
                 "value is probably wrong: compare each literal with the example values in the schema "
                 "(spelling, case, date format) and check every join condition. If you are sure the "
                 "result really is empty, return the same query. Return only the ```sql block.")
