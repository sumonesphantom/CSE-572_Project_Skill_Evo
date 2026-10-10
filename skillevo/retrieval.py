"""BM25 retrieval of skills for a task (no external dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter

from .skills import Skill

_STOP = set("""a an the of in on for to and or is are was were be by with what which who how many
much does do did from that this as at it its list give show find name all each their there than""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z][a-z0-9_]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP:
            continue
        out.append(t)
        out.extend(p for p in t.split("_") if p and p != t and p not in _STOP)
    return out


def task_query(task) -> str:
    return f"{task.question} {task.evidence}"


def retrieve(skills: list[Skill], query: str, k: int = 3, min_score: float = 0.5,
             k1: float = 1.2, b: float = 0.75) -> list[Skill]:
    if not skills or k <= 0:
        return []
    docs = [tokenize(f"{s.name} {s.description} {' '.join(s.tags)}") for s in skills]
    avgdl = sum(len(d) for d in docs) / len(docs) or 1.0
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    q = tokenize(query)
    scored = []
    for s, d in zip(skills, docs):
        tf = Counter(d)
        score = 0.0
        for t in set(q):
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        if score >= min_score:
            scored.append((score, s.id, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored[:k]]
