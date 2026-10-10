"""The task agent: question + schema + retrieved skills -> SQL, with one repair turn."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .llm import extract_sql
from .retrieval import retrieve, task_query
from .skills import Skill, SkillLibrary
from .sqlexec import is_correct, schema_text

SYSTEM = (
    "You are an expert SQLite analyst. Write ONE SQLite query that answers the question. "
    "Return only the query inside a ```sql code block."
)


@dataclass
class Trajectory:
    task_id: str
    skills: list[str]            # version keys of skills in context
    sql: str
    correct: bool
    error: str | None
    n_rows: int
    repaired: bool = False
    first_sql: str | None = None
    first_error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def build_prompt(task, skills: list[Skill], schema: str) -> list[dict]:
    parts = [f"Database schema:\n{schema}"]
    if skills:
        guide = "\n\n".join(s.text() for s in skills)
        parts.append("Skills that may help (follow them when relevant):\n" + guide)
    if task.evidence:
        parts.append(f"External knowledge: {task.evidence}")
    parts.append(f"Question: {task.question}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


class Solver:
    def __init__(self, llm, top_k: int = 3, min_score: float = 0.5, repair: bool = True,
                 sql_timeout: float = 30.0, schema_rows: int = 2):
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.repair = repair
        self.sql_timeout = sql_timeout
        self.schema_rows = schema_rows

    def select_skills(self, task, library: SkillLibrary) -> list[Skill]:
        return retrieve(library.active(), task_query(task), self.top_k, self.min_score)

    def solve(self, task, library: SkillLibrary | None = None, skills: list[Skill] | None = None) -> Trajectory:
        if skills is None:
            skills = self.select_skills(task, library) if library is not None else []
        schema = schema_text(task.db_path, self.schema_rows)
        msgs = build_prompt(task, skills, schema)
        reply = self.llm.chat(msgs)
        sql = extract_sql(reply)
        ok, err, n = is_correct(task.db_path, sql, task.gold_sql, self.sql_timeout)
        traj = Trajectory(task.id, [s.key for s in skills], sql, ok, err, n)
        if self.repair and err:
            # Execution feedback only (no gold), as a deployed agent would see.
            msgs = msgs + [{"role": "assistant", "content": reply},
                           {"role": "user", "content": f"That query failed with: {err}\nFix it. Return only the corrected ```sql block."}]
            sql2 = extract_sql(self.llm.chat(msgs))
            ok2, err2, n2 = is_correct(task.db_path, sql2, task.gold_sql, self.sql_timeout)
            traj = Trajectory(task.id, traj.skills, sql2, ok2, err2, n2, True, sql, err)
        return traj
