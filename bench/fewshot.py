"""Few-shot demonstration pool + BM25 retrieval over questions.

Pool per target dataset (never contains val/test questions):
- bird   : our BIRD mini-dev *train* split (247 q's, same 11 DBs, disjoint questions)
- spider : our Spider *train* split (309 q's from dev DBs) + official Spider train (7,000 q's, other DBs)

Demonstrations are shown as (question, evidence, SQL) without their schema, to keep the
context short; examples from the same database are ranked first because they share the
target schema's tables and value conventions.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from skillevo.retrieval import tokenize


@dataclass
class Demo:
    id: str
    db_id: str
    question: str
    evidence: str
    sql: str


class DemoPool:
    def __init__(self, demos: list[Demo], k1: float = 1.2, b: float = 0.75):
        self.demos = demos
        self.docs = [tokenize(f"{d.question} {d.evidence}") for d in demos]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1) or 1.0
        self.df = Counter(t for d in self.docs for t in set(d))
        self.tfs = [Counter(d) for d in self.docs]
        self.k1, self.b = k1, b

    def top(self, task, k: int = 3, same_db_bonus: float = 2.0) -> list[Demo]:
        q = set(tokenize(f"{task.question} {task.evidence}"))
        n = len(self.demos)
        scored = []
        for i, (d, tf, doc) in enumerate(zip(self.demos, self.tfs, self.docs)):
            if d.id == task.id:
                continue
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            if d.db_id == task.db_id:
                s += same_db_bonus
            scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.demos[i] for _, i in scored[:k]]


def build_pools(tasks: dict, splits: dict, spider_train_json: str | Path | None) -> dict[str, DemoPool]:
    pools = {}
    for ds in ("bird", "spider"):
        demos = [Demo(t.id, t.db_id, t.question, t.evidence, t.gold_sql)
                 for tid in splits.get(ds, {}).get("train", []) if (t := tasks.get(tid))]
        if ds == "spider" and spider_train_json and Path(spider_train_json).exists():
            for i, x in enumerate(json.loads(Path(spider_train_json).read_text())):
                demos.append(Demo(f"spider-train-{i}", x["db_id"], x["question"], "", x["query"]))
        pools[ds] = DemoPool(demos)
    return pools


def format_demos(demos: list[Demo]) -> str:
    out = []
    for d in demos:
        s = f"Question: {d.question}"
        if d.evidence:
            s += f"\nExternal knowledge: {d.evidence}"
        out.append(s + f"\n```sql\n{d.sql.strip()}\n```")
    return "\n\n".join(out)
