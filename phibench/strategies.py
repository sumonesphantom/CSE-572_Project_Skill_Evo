"""Prompt strategies. Phase A is the baseline ladder; each rung adds one thing:

  a1_repo     the repo prompt (DDL + 2 sample rows, 6,000-char cap) - identical to the SkillEvo solver
  a2_mschema  compact schema: typed columns, PKs, 3 example values per column, foreign keys
  a3_rules    a2 + output guidelines
  a4_fs3      a3 + 3 retrieved solved examples (same database first)

Every strategy keeps one repair turn when the query raises an SQLite error.
Phase B strategies are added here as they are implemented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from skillevo.solver import SYSTEM as REPO_SYSTEM
from skillevo.solver import build_prompt as repo_prompt

from .examples import format_demos
from .schema import render

GUIDELINES = """Guidelines:
1. Select only the columns the question asks for, in the order it asks for them.
2. When External knowledge defines a formula, mapping or threshold, apply it exactly as written.
3. Wrap table or column names that contain spaces or special characters in backticks.
4. Before dividing to get a ratio or percentage, CAST the numerator AS REAL.
5. Add ORDER BY ... LIMIT only when the question asks for the highest, lowest, first or last item(s)."""


@dataclass(frozen=True)
class Strategy:
    name: str
    schema: str = "repo"
    rules: bool = False
    k_shots: int = 0
    retriever: str = "bm25_samedb"
    repair: str = "error"          # none | error
    max_tokens: int = 512

    def to_dict(self):
        return asdict(self)


STRATEGIES = {s.name: s for s in [
    Strategy("a1_repo"),
    Strategy("a2_mschema", schema="mschema"),
    Strategy("a3_rules", schema="mschema", rules=True),
    Strategy("a4_fs3", schema="mschema", rules=True, k_shots=3),
]}


def build_messages(task, strat: Strategy, demos=()) -> list[dict]:
    schema = render(task.db_path, strat.schema)
    if strat.schema == "repo" and not strat.rules and not demos:
        return repo_prompt(task, [], schema)
    parts = [f"Database schema:\n{schema}"]
    if demos:
        parts.append("Solved examples from similar questions:\n\n" + format_demos(list(demos)))
    if strat.rules:
        parts.append(GUIDELINES)
    if task.evidence:
        parts.append(f"External knowledge: {task.evidence}")
    parts.append(f"Question: {task.question}")
    return [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
