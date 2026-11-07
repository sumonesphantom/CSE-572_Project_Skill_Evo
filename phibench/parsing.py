"""Pull one SQL statement out of a model reply.

Phi-4-mini is a general instruct model, so replies are less tidy than a SQL-tuned model's:
it may skip the code fence, leave a fence unterminated, add an explanation after the query,
or emit several statements. Order of preference: last ```sql block, last untagged block,
an unterminated fence, then the first SELECT/WITH in plain text.
"""
from __future__ import annotations

import re

from .llm import strip_think

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_-]*)[ \t]*\n?(.*?)```", re.S)
_START = re.compile(r"\b(WITH|SELECT)\b", re.I)


def first_statement(sql: str) -> str:
    """Cut at the first ';' that is outside quotes/backticks/brackets."""
    quote = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote or (quote == "[" and ch == "]"):
                quote = None
        elif ch in "'\"`[":
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql


def _drop_trailing_prose(sql: str) -> str:
    """Unfenced replies often continue with a paragraph after a blank line."""
    parts = re.split(r"\n\s*\n", sql, maxsplit=1)
    if len(parts) == 2 and not re.match(r"\s*(--|/\*|\(|\)|SELECT|WITH|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|JOIN|INNER|LEFT|AND|OR)\b",
                                       parts[1], re.I):
        return parts[0]
    return sql


def extract_sql(text: str) -> str:
    text = strip_think(text)
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text)]
    sql_blocks = [b for lang, b in blocks if lang in ("sql", "sqlite")]
    plain = [b for lang, b in blocks if lang == ""]
    if sql_blocks:
        sql = sql_blocks[-1]
    elif plain:
        sql = plain[-1]
    elif blocks:
        sql = blocks[-1][1]
    else:
        m = re.search(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*)$", text, re.S | re.I)
        if m:
            sql = m.group(1)
        else:
            m = _START.search(text)
            sql = _drop_trailing_prose(text[m.start():]) if m else text
    sql = first_statement(sql.strip())
    return sql.strip().rstrip(";").strip()
