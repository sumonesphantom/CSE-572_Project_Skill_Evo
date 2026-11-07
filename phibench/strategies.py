"""Prompt strategies. Phase A is the baseline ladder; each rung adds one thing:

  a1_repo     the repo prompt (DDL + 2 sample rows, 6,000-char cap) - identical to the SkillEvo solver
  a2_mschema  compact schema: typed columns, PKs, 3 example values per column, foreign keys
  a3_rules    a2 + output guidelines
  a4_fs3      a3 + 3 retrieved solved examples (same database first)

Every A strategy keeps one repair turn when the query raises an SQLite error.
Phase B (each tested on bird_val against its A parent):
  b1_rules / b1_fs3   a3 / a4 + schema-aware repair (phibench.repair), up to 2 rounds, retry on empty
  b2_fs3              b1_fs3 + value retrieval (phibench.values): exact stored spellings in the prompt
                      and in the empty-result retry
  b2m_fs3             b2_fs3 with only the longest match per question span
  b2r_fs3             b1_fs3 + stored values in the empty-result retry only (prompt identical to b1_fs3)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from skillevo.solver import SYSTEM as REPO_SYSTEM
from skillevo.solver import build_prompt as repo_prompt

from .examples import format_demos
from .schema import render
from .values import value_hints

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
    repair: str = "error"          # none | error (SQLite message only) | schema (B1: database-grounded hints)
    repair_rounds: int = 1         # max repair turns
    retry_empty: bool = False      # B1: also retry when the query runs but returns no rows
    value_hints: bool = False      # B2: stored values matching the question, in the prompt and the empty retry
    value_mode: str = "all"        # all | maximal (longest match per question span only)
    value_retry_only: bool = False # B2 variant: values only in the empty-result retry, not in the prompt
    max_tokens: int = 512

    def to_dict(self):
        return asdict(self)


STRATEGIES = {s.name: s for s in [
    Strategy("a1_repo"),
    Strategy("a2_mschema", schema="mschema"),
    Strategy("a3_rules", schema="mschema", rules=True),
    Strategy("a4_fs3", schema="mschema", rules=True, k_shots=3),
    # B1: schema-aware repair, up to 2 rounds, plus a retry on empty results (on both tied A rungs)
    Strategy("b1_rules", schema="mschema", rules=True, repair="schema", repair_rounds=2, retry_empty=True),
    Strategy("b1_fs3", schema="mschema", rules=True, k_shots=3, repair="schema", repair_rounds=2, retry_empty=True),
    # B2: value retrieval on top of the B1 winner
    Strategy("b2_fs3", schema="mschema", rules=True, k_shots=3, repair="schema", repair_rounds=2, retry_empty=True,
             value_hints=True),
    # after b2_fs3 lost 10 / gained 8 on bird_val (partial matches such as 'Serra' next to 'Serra Angel'):
    Strategy("b2m_fs3", schema="mschema", rules=True, k_shots=3, repair="schema", repair_rounds=2, retry_empty=True,
             value_hints=True, value_mode="maximal"),
    Strategy("b2r_fs3", schema="mschema", rules=True, k_shots=3, repair="schema", repair_rounds=2, retry_empty=True,
             value_hints=True, value_retry_only=True),
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
    if strat.value_hints and not strat.value_retry_only:
        hints = value_hints(task, maximal=strat.value_mode == "maximal")
        if hints:
            parts.append(hints)
    if task.evidence:
        parts.append(f"External knowledge: {task.evidence}")
    parts.append(f"Question: {task.question}")
    return [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
