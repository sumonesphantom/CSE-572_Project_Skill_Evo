"""Task loading and train / val / test splitting for BIRD mini-dev and Spider."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Task:
    id: str
    dataset: str
    db_id: str
    db_path: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str = "unknown"

    def to_dict(self):
        return asdict(self)


def load_bird(root: str | Path) -> list[Task]:
    root = Path(root)
    items = json.loads((root / "mini_dev_sqlite.json").read_text())
    tasks = []
    for x in items:
        db = root / "dev_databases" / x["db_id"] / f"{x['db_id']}.sqlite"
        tasks.append(Task(
            id=f"bird-{x['question_id']}", dataset="bird", db_id=x["db_id"], db_path=str(db),
            question=x["question"], evidence=x.get("evidence", "") or "",
            gold_sql=x["SQL"], difficulty=x.get("difficulty", "unknown"),
        ))
    return tasks


def _spider_hardness(sql: str) -> str:
    s = sql.upper()
    score = sum(s.count(k) for k in ("JOIN", "GROUP BY", "ORDER BY", "HAVING")) \
        + 2 * sum(s.count(k) for k in ("INTERSECT", "UNION", "EXCEPT")) + 2 * max(s.count("SELECT") - 1, 0)
    return "simple" if score <= 1 else "moderate" if score <= 3 else "challenging"


def load_spider(root: str | Path) -> list[Task]:
    root = Path(root)
    items = json.loads((root / "dev.json").read_text())
    tasks = []
    for i, x in enumerate(items):
        db = root / "database" / x["db_id"] / f"{x['db_id']}.sqlite"
        tasks.append(Task(
            id=f"spider-{i}", dataset="spider", db_id=x["db_id"], db_path=str(db),
            question=x["question"], evidence="", gold_sql=x["query"],
            difficulty=_spider_hardness(x["query"]),
        ))
    return tasks


def stratified_split(tasks: list[Task], fractions: dict[str, float], seed: int = 0) -> dict[str, list[str]]:
    """Split by (db_id, difficulty) strata so every split sees every database."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for t in tasks:
        strata[(t.db_id, t.difficulty)].append(t.id)
    names = list(fractions)
    out = {n: [] for n in names}
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        n = len(ids)
        start = 0
        for k, name in enumerate(names):
            end = n if k == len(names) - 1 else start + round(fractions[name] * n)
            out[name].extend(ids[start:end])
            start = end
    return out


def save_tasks(tasks: list[Task], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_tasks(path: str | Path) -> dict[str, Task]:
    tasks = {}
    with open(path) as f:
        for line in f:
            t = Task(**json.loads(line))
            tasks[t.id] = t
    return tasks
