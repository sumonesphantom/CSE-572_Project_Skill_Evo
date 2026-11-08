"""Skill evolver: turns a batch of trajectories into lifecycle operations.

The evolver LLM sees failures (with execution feedback and, by default, the
gold SQL for training tasks), a few successes, and the current library. It
returns a JSON list of operations: create / update / merge / discard.
When `show_stats` is on (guarded arm), the per-skill lift measured by the
monitor is shown too, so the evolver can target harmful or useless skills.
"""
from __future__ import annotations

import json
import random

from .llm import extract_json

SYSTEM = """You maintain a library of SKILLS for a Text-to-SQL agent (SQLite).
A skill is short, reusable, database-agnostic guidance that fixes a recurring mistake,
e.g. how to compute ratios/percentages, handle dates stored as text, pick DISTINCT,
tie-break ORDER BY ... LIMIT 1, or apply the 'External knowledge' formula literally.
Good skills are general (not about one question), concrete (show a SQL pattern), and short (<= 120 words).

You may propose these operations:
- {"op": "create", "name": ..., "description": "when to use, with trigger keywords", "content": "the procedure + a tiny SQL pattern", "tags": [keywords], "rationale": ...}
- {"op": "update", "skill_id": "S00x", "description": optional new text, "content": "full revised content", "rationale": ...}
- {"op": "merge", "skill_ids": ["S00x", "S00y"], "name": ..., "description": ..., "content": ..., "tags": [...], "rationale": ...}
- {"op": "discard", "skill_id": "S00x", "rationale": ...}

Rules: prefer updating an existing skill over creating a near-duplicate; merge skills that overlap;
discard skills that are wrong, too specific, or (if statistics are shown) have negative lift.
Return ONLY a JSON array of at most MAX_OPS operations. Return [] if nothing should change."""

DB_SKILLS = """
Skills may also be tied to ONE database. A database skill names the exact tables, join path and columns
for a recurring kind of question on that database, with one worked SQL example on those tables. Add
"db": "<database name>" (as shown in the episodes) to a create or merge op to make it a database skill.
Prefer a database skill when a failure comes from picking the wrong table, join or column; keep general
skills for mistakes that recur across databases."""


def _fmt_rejected(r: dict) -> str:
    what = r.get("summary", "?")
    verdict = f"verdict {r.get('label', '?')}: fixed {r.get('gained', 0)}, broke {r.get('lost', 0)}"
    body = (r.get("content") or "").replace("\n", " ")
    return f"- {what} ({verdict})" + (f": {body[:160]}" if body else "")


def _fmt_traj(task, traj, show_gold: bool, show_db: bool = False) -> str:
    lines = ([f"Database: {task.db_id}"] if show_db else []) + [f"Q: {task.question}"]
    if task.evidence:
        lines.append(f"External knowledge: {task.evidence}")
    lines.append(f"Skills used: {', '.join(traj['skills']) or 'none'}")
    lines.append(f"Agent SQL: {traj['sql']}")
    lines.append(f"Outcome: {'ERROR: ' + traj['error'] if traj['error'] else ('correct' if traj['correct'] else 'wrong result')}")
    if show_gold and not traj["correct"]:
        lines.append(f"Reference SQL: {task.gold_sql}")
    return "\n".join(lines)


class Evolver:
    def __init__(self, llm, max_ops: int = 4, n_fail: int = 10, n_success: int = 3,
                 show_gold: bool = True, show_stats: bool = False,
                 allowed_ops=("create", "update", "merge", "discard"), seed: int = 0,
                 show_rejections: bool = False, db_skills: bool = False):
        self.llm = llm
        self.max_ops = max_ops
        self.n_fail = n_fail
        self.n_success = n_success
        self.show_gold = show_gold
        self.show_stats = show_stats
        self.allowed_ops = set(allowed_ops)
        self.rng = random.Random(seed)
        self.show_rejections = show_rejections   # fix C1: show earlier rejected proposals and their verdicts
        self.db_skills = db_skills               # fix C2: allow skills tied to one database

    def build_messages(self, library, trajs, tasks, skill_stats=None, rejected=None):
        fails = [t for t in trajs if not t["correct"]]
        wins = [t for t in trajs if t["correct"]]
        self.rng.shuffle(fails)
        self.rng.shuffle(wins)
        lib_lines = []
        for s in library.active():
            db = next((t[3:] for t in s.tags if isinstance(t, str) and t.startswith("db:")), None)
            scope = f" (database {db})" if self.db_skills and db else ""
            line = f"- {s.id} (v{s.version}) [{s.name}]{scope} when: {s.description}\n  content: {s.content}"
            if self.show_stats and skill_stats and s.id in skill_stats:
                st = skill_stats[s.id]
                line += f"\n  stats: used {st['n']}x, lift vs no-skill {st['lift']:+.2f} (95% CI {st['ci_lo']:+.2f}..{st['ci_hi']:+.2f}), health={st['health']}"
            lib_lines.append(line)
        body = [
            "CURRENT LIBRARY:\n" + ("\n".join(lib_lines) if lib_lines else "(empty)"),
            "FAILED EPISODES:\n" + "\n\n".join(_fmt_traj(tasks[t["task_id"]], t, self.show_gold, self.db_skills)
                                                for t in fails[: self.n_fail]),
        ]
        if self.show_rejections and rejected:
            body.insert(1, "PREVIOUSLY REJECTED PROPOSALS (each was tested on held-out questions and did not help; "
                           "do not propose them or close variants again - try a different, more specific fix):\n"
                        + "\n".join(_fmt_rejected(r) for r in rejected))
        if wins and self.n_success:
            body.append("SUCCESSFUL EPISODES:\n" + "\n\n".join(_fmt_traj(tasks[t["task_id"]], t, False, self.db_skills)
                                                    for t in wins[: self.n_success]))
        allowed = ", ".join(sorted(self.allowed_ops))
        body.append(f"Allowed operations this round: {allowed}. Propose at most {self.max_ops}. JSON array only.")
        system = SYSTEM.replace("MAX_OPS", str(self.max_ops)) + (DB_SKILLS if self.db_skills else "")
        return [{"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(body)}]

    def propose(self, library, trajs, tasks, skill_stats=None, rejected=None) -> tuple[list[dict], str]:
        msgs = self.build_messages(library, trajs, tasks, skill_stats, rejected)
        raw = self.llm.chat(msgs)
        try:
            ops = extract_json(raw)
        except ValueError:
            return [], raw
        if isinstance(ops, dict):
            ops = ops.get("operations", [ops])
        ops = [o for o in (validate_op(o, library, self.allowed_ops) for o in ops) if o][: self.max_ops]
        dbs = {t.db_id for t in tasks.values()}
        for o in ops:
            db = o.pop("db", None)
            if self.db_skills and o["op"] in ("create", "merge") and isinstance(db, str) and db in dbs:
                o["tags"] = [t for t in o["tags"] if not t.startswith("db:")] + [f"db:{db}"]
            elif not self.db_skills:
                o["tags"] = [t for t in o["tags"] if not t.startswith("db:")]
        return ops, raw


def validate_op(op, library, allowed) -> dict | None:
    if not isinstance(op, dict) or op.get("op") not in allowed:
        return None
    kind = op["op"]
    if kind == "create":
        if not op.get("name") or not op.get("content"):
            return None
        op.setdefault("description", op["name"])
    elif kind in ("update", "discard"):
        if library.get(op.get("skill_id")) is None:
            return None
        if kind == "update" and not (op.get("content") or op.get("description")):
            return None
    elif kind == "merge":
        ids = [i for i in op.get("skill_ids", []) if library.get(i) is not None]
        if len(set(ids)) < 2 or not op.get("content"):
            return None
        op["skill_ids"] = list(dict.fromkeys(ids))
        op.setdefault("name", "merged skill")
        op.setdefault("description", op["name"])
    tags = op.get("tags", [])
    op["tags"] = [str(t) for t in tags] if isinstance(tags, list) else []
    return op


def apply_op(library, op, round_):
    """Apply an op to `library` in place. Returns (touched_skill_ids_before, new_skill_or_None)."""
    kind = op["op"]
    if kind == "create":
        s = library.create(op["name"], op["description"], op["content"], op["tags"], round_)
        return [], s
    if kind == "update":
        s = library.update(op["skill_id"], round_, description=op.get("description"),
                           content=op.get("content"), tags=op.get("tags") or None)
        return [op["skill_id"]], s
    if kind == "merge":
        s = library.merge(op["skill_ids"], op["name"], op["description"], op["content"], op["tags"], round_)
        return list(op["skill_ids"]), s
    library.discard(op["skill_id"])
    return [op["skill_id"]], None


def describe_op(op) -> str:
    k = op["op"]
    if k == "create":
        return f"create '{op['name']}'"
    if k == "merge":
        return f"merge {'+'.join(op['skill_ids'])} -> '{op['name']}'"
    return f"{k} {op['skill_id']}"


def op_json(op) -> str:
    return json.dumps(op, ensure_ascii=False)
