"""B2: value retrieval. Match phrases from the question against the text values stored in the
database, so the model sees the exact spelling ('Fresno Unified', 'Durability', '201306').

Index per database (built once, cached in memory): every distinct short text value (<= 80 chars,
up to 20,000 per column) of every column, keyed by lower-cased value and by word. No model calls.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
from collections import defaultdict
from functools import lru_cache

from skillevo.retrieval import _STOP

from .schema import quote_ident

MAX_PER_COL = 20_000
MAX_LEN = 80
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._/+-]*")
_EXTRA_STOP = {"please", "name", "names", "number", "list", "total", "average", "percentage", "ratio",
               "highest", "lowest", "most", "least", "top", "between", "among", "whose", "were", "has", "have",
               "had", "not", "also", "only", "into", "per", "over", "under", "more", "less", "than"}


def _words(s: str) -> list[str]:
    return [w.lower().strip("'.") for w in _WORD.findall(s)]


class ValueIndex:
    def __init__(self, db_path: str):
        self.exact: dict[str, list[tuple[str, str, str]]] = defaultdict(list)   # lower value -> (table, col, value)
        self.by_word: dict[str, set[str]] = defaultdict(set)                   # word -> lower values
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for t in tables:
                for col in [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]:
                    try:
                        rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{t}" WHERE typeof("{col}") = \'text\' '
                                            f'AND length("{col}") BETWEEN 1 AND {MAX_LEN} LIMIT {MAX_PER_COL}').fetchall()
                    except sqlite3.Error:
                        continue
                    for (v,) in rows:
                        low = v.strip().lower()
                        if not low:
                            continue
                        self.exact[low].append((t, col, v))
                        ws = _words(low)
                        if 0 < len(ws) <= 6:
                            for w in ws:
                                self.by_word[w].add(low)
        finally:
            conn.close()

    # ----------------------------------------------------------- question -> values
    @staticmethod
    def salient(text: str) -> set[str]:
        """Words that look like names or values: capitalised, quoted, or containing digits."""
        out = set()
        for q in re.findall(r"['\"]([^'\"]{1,80})['\"]", text):
            out.update(_words(q))
        for w in _WORD.findall(text):
            if w[0].isupper() or any(ch.isdigit() for ch in w):
                out.add(w.lower().strip("'."))
        return out - _STOP

    def phrases(self, text: str, max_n: int = 4) -> list[str]:
        quoted = [q.lower() for q in re.findall(r"['\"]([^'\"]{2,80})['\"]", text)]
        sal = self.salient(text)
        ws = _words(text)
        grams = []
        for n in range(max_n, 0, -1):
            for i in range(len(ws) - n + 1):
                g = ws[i:i + n]
                if g[0] in _STOP or g[-1] in _STOP or not any(w in sal for w in g):
                    continue
                if n == 1 and (g[0] in _EXTRA_STOP or len(g[0]) < 3 or g[0].isdigit()):
                    continue
                grams.append(" ".join(g))
        return list(dict.fromkeys(quoted + grams))

    def match_question(self, text: str, max_hits: int = 10, per_col: int = 3) -> list[tuple[str, str, str]]:
        hits: list[tuple[str, str, str]] = []
        seen, per = set(), defaultdict(int)

        def add(t, c, v):
            if (t, c, v) in seen or per[(t, c)] >= per_col:
                return
            seen.add((t, c, v))
            per[(t, c)] += 1
            hits.append((t, c, v))

        phrases = self.phrases(text)
        for p in phrases:                       # 1) whole value equals a phrase (longest phrases first)
            for t, c, v in self.exact.get(p, []):
                add(t, c, v)
        for p in phrases:                       # 2) a stored value contains all words of a phrase
            ws = p.split()
            if len(ws) == 1 and len(ws[0]) < 4:
                continue
            cands = set.intersection(*(self.by_word.get(w, set()) for w in ws)) if ws else set()
            if not cands or len(cands) > 5:     # too generic to be a useful hint
                continue
            for low in sorted(cands, key=len):
                for t, c, v in self.exact[low]:
                    add(t, c, v)
        return hits[:max_hits]

    # ----------------------------------------------------------- literal -> stored values
    def stored_like(self, literal: str, n: int = 4) -> list[tuple[str, str, str]]:
        low = literal.strip().lower()
        if low in self.exact:
            return self.exact[low][:n]
        ws = _words(low)
        cands = set.intersection(*(self.by_word.get(w, set()) for w in ws)) if ws else set()
        if 0 < len(cands) <= 50:
            out = sorted(cands, key=len)[:n]
        else:
            out = difflib.get_close_matches(low, list(self.exact), n=n, cutoff=0.8)
        return [h for low2 in out for h in self.exact[low2]][:n]

    def is_stored(self, literal: str) -> bool:
        return any(v == literal for _, _, v in self.exact.get(literal.strip().lower(), []))


@lru_cache(maxsize=64)
def index_for(db_path: str) -> ValueIndex:
    return ValueIndex(db_path)


def value_hints(task) -> str:
    hits = index_for(task.db_path).match_question(f"{task.question} {task.evidence}")
    if not hits:
        return ""
    lines = [f"- {quote_ident(t)}.{quote_ident(c)} = '{v.replace(chr(39), chr(39) * 2)}'" for t, c, v in hits]
    return "Stored values that match words in the question (use these exact spellings):\n" + "\n".join(lines)


def empty_value_hints(db_path: str, sql: str) -> str:
    """For an empty result: literals in the query that are not stored as written, with stored look-alikes."""
    idx = index_for(db_path)
    out = []
    for lit in dict.fromkeys(re.findall(r"'((?:[^']|'')*)'", sql)):
        lit = lit.replace("''", "'")
        if not lit or len(lit) > MAX_LEN or lit.replace(".", "").replace("-", "").isdigit() or "%" in lit:
            continue
        if idx.is_stored(lit):
            continue
        near = [h for h in idx.stored_like(lit) if h[2] != lit]
        if near:
            shown = ", ".join(f"{quote_ident(t)}.{quote_ident(c)} = '{v}'" for t, c, v in near)
            out.append(f"'{lit}' is not stored exactly like that. Stored values that look similar: {shown}.")
    return "\n".join(out)
