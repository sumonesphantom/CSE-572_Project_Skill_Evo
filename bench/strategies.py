"""Prompt-engineering ladder. Each rung changes one thing relative to the one before it,
so the per-rung delta in the results table is attributable to that change.

  P0 zs_ddl        zero-shot, raw CREATE TABLE only
  P1 zs_repo       repo default: DDL + 2 sample rows (identical to skillevo.solver.build_prompt;
                   shares the LLM cache with SkillEvo's no-skill baseline)
  P1b zs_repo_full control: P1 without the 6,000-char schema cap (20,000 like P2), so the
                   P1 -> P2 step can be split into "more schema visible" and "M-Schema format"
  P2 zs_mschema    compact M-Schema: typed columns, PK, 3 distinct example values, FK lines
  P3 zs_desc       P2 + BIRD column/value descriptions (no-op on Spider)
  P4 zs_rules      P3 + output-format rules (exact columns, evidence formulas, CAST for ratios)
  P5 fs3           P4 + 3 retrieved demonstrations (BM25, same-DB first, train split only)
  P6 fs3_cot       P5 + brief step-by-step reasoning before the SQL
  P7 fs3_sc5       P5 + self-consistency: greedy + 4 samples (T=0.7), execution-result majority vote
  P8 fs3_cot_sc5   P6 + the same self-consistency vote
  P4m/P5m/P7m      ms_rules / ms_fs3 / ms_fs3_sc5: P4 / P5 / P7 built on P2 (no descriptions), added
                   after bird_val showed P3's descriptions cost ~7 EX points on Coder-7B
Every rung keeps the repo's one-shot repair turn (SQLite error -> retry); first-attempt EX is
logged too, so the effect of repair is measured for free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from skillevo.solver import SYSTEM as REPO_SYSTEM
from skillevo.solver import build_prompt as repo_prompt

from .fewshot import format_demos
from .schema import render

RULES = """Rules:
- Return exactly the columns the question asks for, in that order; no extra columns.
- If External knowledge gives a formula, value mapping or threshold, use it literally.
- Quote identifiers that contain spaces or symbols with backticks, e.g. `Free Meal Count (K-12)`.
- For ratios and percentages, CAST the numerator AS REAL before dividing.
- Use ORDER BY ... LIMIT only when the question asks for a top/bottom/most/least item."""

COT_SYSTEM = (
    "You are an expert SQLite analyst. First think briefly in at most 5 short numbered steps: "
    "the tables and columns needed, join keys, filters (copy exact values/formulas from External "
    "knowledge), aggregation, and the output columns. Then give ONE SQLite query in a ```sql code "
    "block at the very end."
)


@dataclass(frozen=True)
class Strategy:
    name: str
    schema: str = "ddl_samples"
    rules: bool = False
    k_shots: int = 0
    cot: bool = False
    n_samples: int = 1          # >1 => self-consistency with execution voting
    temperature: float = 0.7    # for samples 1..n-1 (sample 0 is greedy)
    max_tokens: int = 512

    def to_dict(self):
        return asdict(self)


LADDER = {s.name: s for s in [
    Strategy("zs_ddl", schema="ddl"),
    Strategy("zs_repo", schema="ddl_samples"),
    Strategy("zs_repo_full", schema="ddl_samples_full"),
    Strategy("zs_mschema", schema="mschema"),
    Strategy("zs_desc", schema="mschema_desc"),
    Strategy("zs_rules", schema="mschema_desc", rules=True),
    Strategy("fs3", schema="mschema_desc", rules=True, k_shots=3),
    Strategy("fs3_cot", schema="mschema_desc", rules=True, k_shots=3, cot=True, max_tokens=1024),
    Strategy("fs3_sc5", schema="mschema_desc", rules=True, k_shots=3, n_samples=5),
    Strategy("fs3_cot_sc5", schema="mschema_desc", rules=True, k_shots=3, cot=True, n_samples=5, max_tokens=1024),
    # branch added after the bird_val ladder showed descriptions (P3) hurt: same rungs on P2's schema
    Strategy("ms_rules", schema="mschema", rules=True),
    Strategy("ms_fs3", schema="mschema", rules=True, k_shots=3),
    Strategy("ms_fs3_sc5", schema="mschema", rules=True, k_shots=3, n_samples=5),
]}
BRANCH = {"zs_repo_full": "P1b", "ms_rules": "P4m", "ms_fs3": "P5m", "ms_fs3_sc5": "P7m"}
_MAIN = [n for n in LADDER if n not in BRANCH]
RUNG = {**{name: f"P{i}" for i, name in enumerate(_MAIN)}, **BRANCH}


def build_messages(task, strat: Strategy, skills=(), demos=()) -> list[dict]:
    schema = render(task.db_path, strat.schema)
    if strat.schema == "ddl_samples" and not (strat.rules or demos or strat.cot):
        return repo_prompt(task, list(skills), schema)  # byte-identical to the repo solver prompt
    parts = [f"Database schema:\n{schema}"]
    if skills:
        parts.append("Skills that may help (follow them when relevant):\n" + "\n\n".join(s.text() for s in skills))
    if demos:
        parts.append("Solved examples from similar questions:\n" + format_demos(demos))
    if strat.rules:
        parts.append(RULES)
    if task.evidence:
        parts.append(f"External knowledge: {task.evidence}")
    parts.append(f"Question: {task.question}")
    return [{"role": "system", "content": COT_SYSTEM if strat.cot else REPO_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]
