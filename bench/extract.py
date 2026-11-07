"""Robust SQL extraction (SmolLM2 rungs S1+). Ported unchanged from bench_llama/extract.py (charan-llama).

Small models often leave a code fence unclosed, prepend prose, or append explanations;
the repo's extractor then passes the fence itself to SQLite."""
from __future__ import annotations

import re

_START = re.compile(r"\b(SELECT|WITH)\b", re.I)


def extract_sql(text: str) -> str:
    closed = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    if closed:
        body = closed[0]
    else:
        body = re.sub(r"```(?:sql|sqlite)?", "", text, flags=re.I)   # unclosed / stray fences
    m = _START.search(body)
    if m:
        body = body[m.start():]
    body = body.split(";")[0]                                           # first statement only
    # drop trailing prose lines that are clearly not SQL (e.g. "This query ...")
    lines = [l for l in body.splitlines()]
    while lines and re.match(r"^\s*(this|the|note|here|explanation)\b", lines[-1], re.I):
        lines.pop()
    return "\n".join(lines).strip()
