"""Improved solver for small models: compact/pruned schema, BM25 few-shot examples from the
TRAIN split, robust extraction, execution-guided self-consistency, optional repair."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from skillevo.sqlexec import execute, is_correct
from skillevo.sqlexec import schema_text as repo_schema_text

from .extract import extract_sql
from .schema import schema_for, toks

SYSTEM = ("You are an expert SQLite developer. Write ONE SQLite query that answers the question. "
          "Use only the tables and columns in the schema, and copy column names exactly (with backticks "
          "if they contain spaces). Return only the query in a ```sql code block.")


@dataclass
class Variant:
    name: str
    schema: str = "repo"         # repo | compact | pruned
    prune: int = 4
    descriptions: bool = True
    values: bool = True
    fewshot: int = 0
    samples: int = 1             # >1 = self-consistency
    temperature: float = 0.8
    repair: bool = True
    robust_extract: bool = True
    prompt: str = "repo"         # repo | improved
    fast: bool = False           # cap length + stop when prose follows the closed ```sql block


# Stops only fire AFTER a closed code block or on explanation openers, never on the opening fence.
FAST_STOP = ["```\n\n", "\n\nThis query", "\n\nExplanation", "\n\nNote:", "\n\nThis SQL"]
FAST_TOKENS = 300


class FewShot:
    """BM25 over training questions of the same dataset; same-database examples get a bonus."""

    def __init__(self, train_tasks):
        self.items = train_tasks
        self.docs = [Counter(toks(f"{t.question} {t.evidence}")) for t in train_tasks]
        self.df = Counter(w for d in self.docs for w in d)
        self.avg = sum(sum(d.values()) for d in self.docs) / max(len(self.docs), 1)

    def top(self, task, k):
        qt = toks(f"{task.question} {task.evidence}")
        n = len(self.docs)
        scored = []
        for t, d in zip(self.items, self.docs):
            if t.id == task.id or t.dataset != task.dataset:
                continue
            L = sum(d.values())
            s = sum(math.log(1 + (n - self.df[w] + .5) / (self.df[w] + .5)) * d[w] * 2.2 / (d[w] + 1.2 * (.25 + .75 * L / self.avg))
                    for w in qt if w in d)
            if t.db_id == task.db_id:
                s += 2.0
            scored.append((s, t.id, t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [t for _, _, t in scored[:k]]


def build_messages(task, v: Variant, fewshot: FewShot | None):
    if v.schema == "repo":
        schema = repo_schema_text(task.db_path, 2)
    else:
        schema = schema_for(task, v.prune if v.schema == "pruned" else 0, v.descriptions, v.values)
    if v.prompt == "repo":                                    # the team repo's prompt, unchanged
        sys_ = ("You are an expert SQLite analyst. Write ONE SQLite query that answers the question. "
                "Return only the query inside a ```sql code block.")
        parts = [f"Database schema:\n{schema}"]
        if task.evidence:
            parts.append(f"External knowledge: {task.evidence}")
        parts.append(f"Question: {task.question}")
        return [{"role": "system", "content": sys_}, {"role": "user", "content": "\n\n".join(parts)}]
    parts = [f"Database schema:\n{schema}"]
    if fewshot and v.fewshot:
        ex = fewshot.top(task, v.fewshot)
        parts.append("Examples of questions and correct SQL:\n\n" + "\n\n".join(
            f"Question: {e.question}" + (f"\nHint: {e.evidence}" if e.evidence else "") + f"\nSQL: {e.gold_sql.strip()}"
            for e in ex))
    q = f"Question: {task.question}"
    if task.evidence:
        q += f"\nHint: {task.evidence}"
    parts.append(q + "\nSQL:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def _gen(v):
    return {"max_tokens": FAST_TOKENS, "stop": FAST_STOP} if v.fast else {}


def _extract(text, v):
    if v.robust_extract:
        return extract_sql(text)
    from skillevo.llm import extract_sql as repo_extract
    return repo_extract(text)


@dataclass
class Result:
    task_id: str
    sql: str
    correct: bool
    error: str | None
    n_candidates: int
    n_executable: int
    calls: int
    gen_secs: float
    repaired: bool = False
    candidates: list = field(default_factory=list)


def solve(task, v: Variant, client, fewshot=None, sql_timeout=30.0) -> Result:
    msgs = build_messages(task, v, fewshot)
    cands, secs, calls = [], 0.0, 0
    for i in range(v.samples):
        temp = 0.0 if i == 0 else v.temperature                # candidate 0 is always greedy
        text, s = client.chat(msgs, temperature=temp, seed=i, **_gen(v))
        secs += s; calls += 1
        sql = _extract(text, v)
        rows, err = execute(task.db_path, sql, sql_timeout)
        cands.append({"sql": sql, "err": err, "rows": None if err else frozenset(rows), "reply": text})
    ok = [c for c in cands if c["err"] is None]
    if ok:                                                     # execution-guided majority vote
        votes = Counter(c["rows"] for c in ok)
        best_rows = max(votes, key=lambda r: (votes[r], -next(i for i, c in enumerate(ok) if c["rows"] == r)))
        chosen = next(c for c in ok if c["rows"] == best_rows)
    else:
        chosen = cands[0]
    repaired = False
    if chosen["err"] and v.repair:
        fix = msgs + [{"role": "assistant", "content": chosen["reply"]},
                      {"role": "user", "content": f"That query failed with: {chosen['err']}\nFix it. Return only the corrected ```sql block."}]
        text, s = client.chat(fix, **_gen(v))
        secs += s; calls += 1
        chosen = {"sql": _extract(text, v), "reply": text}
        repaired = True
    correct, err, _ = is_correct(task.db_path, chosen["sql"], task.gold_sql, sql_timeout)
    return Result(task.id, chosen["sql"], correct, err, len(cands), len(ok), calls, round(secs, 2), repaired,
                  [c["sql"] for c in cands] if v.samples > 1 else [])
