"""Pools of solved training questions for few-shot prompts.

Pools draw only on training data, never on val/test:
- bird   : the repo's bird:train split (247 questions, the same 11 databases as BIRD val/test)
- spider : the repo's spider:train split + the official Spider train set (train_spider.json,
           different databases from Spider dev, which our spider_test comes from)
Retrieval is BM25 over question + external knowledge; same-database examples rank first
when any exist (retriever "bm25_samedb"), or purely by score ("bm25").
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from skillevo.retrieval import tokenize


@dataclass(frozen=True)
class Demo:
    id: str
    dataset: str
    db_id: str
    question: str
    evidence: str
    sql: str
    db_path: str = ""


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / self.n) if self.n else 1.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(query):
            plist = self.post.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class Pool:
    def __init__(self, demos: list[Demo], embedder=None):
        self.demos = demos
        self.index = BM25([tokenize(f"{d.question} {d.evidence}") for d in demos])
        self.embedder = embedder
        self._emb = None

    def _embeddings(self):
        if self._emb is None:
            self._emb = self.embedder.embed([f"{d.question} {d.evidence}".strip() for d in self.demos])
        return self._emb

    def top(self, task, k: int, retriever: str = "bm25_samedb") -> list[Demo]:
        if k <= 0 or not self.demos:
            return []
        if retriever == "embed_samedb":  # B3: cosine similarity of local embeddings, same database first
            if self.embedder is None:
                raise ValueError("embed_samedb needs an embedder")
            q = self.embedder.embed([f"{task.question} {task.evidence}".strip()])[0]
            sims = self._embeddings() @ q
            cand = [i for i in range(len(self.demos)) if self.demos[i].id != task.id]
            key = lambda i: (self.demos[i].db_id != task.db_id, -float(sims[i]), self.demos[i].id)  # noqa: E731
            return [self.demos[i] for i in sorted(cand, key=key)[:k]]
        sc = self.index.scores(tokenize(f"{task.question} {task.evidence}"))
        cand = [i for i in sc if self.demos[i].id != task.id]
        if retriever == "bm25_samedb":
            key = lambda i: (self.demos[i].db_id != task.db_id, -sc[i], self.demos[i].id)  # noqa: E731
        elif retriever == "bm25":
            key = lambda i: (-sc[i], self.demos[i].id)  # noqa: E731
        else:
            raise ValueError(f"unknown retriever {retriever!r}")
        return [self.demos[i] for i in sorted(cand, key=key)[:k]]


def _from_task(t) -> Demo:
    return Demo(t.id, t.dataset, t.db_id, t.question, t.evidence, t.gold_sql, t.db_path)


def build_pools(tasks: dict, splits: dict, spider_train_json: str | Path | None = None,
                eval_ids: set[str] | None = None, embedder=None) -> dict[str, Pool]:
    pools: dict[str, list[Demo]] = {}
    for ds in ("bird", "spider"):
        ids = splits.get(ds, {}).get("train", [])
        pools[ds] = [_from_task(tasks[i]) for i in ids if i in tasks]
    p = Path(spider_train_json) if spider_train_json else None
    if p and p.exists():
        dbdir = p.parent / "database"
        for i, x in enumerate(json.loads(p.read_text())):
            db = dbdir / x["db_id"] / f"{x['db_id']}.sqlite"
            pools["spider"].append(Demo(f"spider-train-{i}", "spider", x["db_id"], x["question"], "", x["query"], str(db)))
    dropped: list[str] = []
    if eval_ids:
        # BIRD mini-dev lists questions 137 and 138 twice; the repo split puts one copy of 138 in
        # train and one in val. Anything that is also an eval task is removed from the pool.
        ev = set(eval_ids)
        for ds in pools:
            dropped += [d.id for d in pools[ds] if d.id in ev]
            pools[ds] = [d for d in pools[ds] if d.id not in ev]
    out = {ds: Pool(v, embedder) for ds, v in pools.items()}
    for p in out.values():
        p.dropped = dropped
    return out


def dealias(sql: str, db_path: str) -> str:
    """B3: rewrite T1/T2-style aliases to the real table names, e.g. `T1.School` -> `schools.School`.
    Only used when each table appears once and the rewritten query returns the same rows."""
    from .repair import aliases, table_columns   # local import: repair imports schema only
    from .schema import quote_ident
    from skillevo.sqlexec import execute
    if not db_path or not Path(db_path).exists():
        return sql
    tables = table_columns(db_path)
    amap = {a: (t, w) for a, (t, w) in aliases(sql, tables).items() if a != t.lower()}
    if not amap:
        return sql
    reals = [t for t, _ in amap.values()]
    if len(reals) != len(set(reals)):            # self-join: aliases are needed
        return sql
    out = sql
    for _, (t, w) in amap.items():
        out = re.sub(rf"(\b(?:FROM|JOIN)\s+[`\"\[]?{re.escape(t)}[`\"\]]?)\s+(?:AS\s+)?{re.escape(w)}\b", r"\1", out, flags=re.I)
        out = re.sub(rf"(?<![\w.`]){re.escape(w)}\.", quote_ident(t) + ".", out)
    if out == sql:
        return sql
    a, ea = execute(db_path, sql, 30)
    b, eb = execute(db_path, out, 30)
    return out if ea is None and eb is None and frozenset(a) == frozenset(b) else sql


def format_demos(demos: list[Demo], dealiased: bool = False) -> str:
    blocks = []
    for n, d in enumerate(demos, 1):
        lines = [f"Example {n} (database: {d.db_id})", f"Question: {d.question}"]
        if d.evidence:
            lines.append(f"External knowledge: {d.evidence}")
        sql = dealias(d.sql, d.db_path) if dealiased else d.sql
        lines.append(f"```sql\n{sql.strip()}\n```")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
