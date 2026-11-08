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


def skill_db(skill: Skill) -> str | None:
    """The database a skill is tied to (tag "db:<id>"), or None for a general skill."""
    for t in skill.tags:
        if isinstance(t, str) and t.startswith("db:"):
            return t[3:]
    return None


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
                 sql_timeout: float = 30.0, schema_rows: int = 2, db_skills: bool = False, max_db_skills: int = 2):
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.repair = repair
        self.sql_timeout = sql_timeout
        self.schema_rows = schema_rows
        self.db_skills = db_skills          # fix C2: skills tagged "db:<id>" are offered to that database's tasks
        self.max_db_skills = max_db_skills

    def select_skills(self, task, library: SkillLibrary) -> list[Skill]:
        skills = library.active()
        if not self.db_skills:
            return retrieve(skills, task_query(task), self.top_k, self.min_score)
        # C2: a database skill is always eligible for its own database (BM25 only orders them) and
        # never shown for another database; general skills fill the rest by the usual BM25 rule.
        scoped = [s for s in skills if skill_db(s) == task.db_id]
        general = [s for s in skills if skill_db(s) is None]
        picked = retrieve(scoped, task_query(task), min(self.max_db_skills, self.top_k), min_score=-1.0)
        return picked + retrieve(general, task_query(task), self.top_k - len(picked), self.min_score)

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
