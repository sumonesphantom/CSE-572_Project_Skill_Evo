"""Offline tests: toy database + scripted fake LLMs (no Ollama needed).

The fake solver writes integer division for ratio questions (wrong) unless a
skill mentioning CAST is in context, and writes a wrong COUNT(DISTINCT ...)
when a 'count' skill is in context. The fake evolver proposes the helpful
ratio skill in round 1 and the harmful count skill in round 2, so we know the
ground truth the monitor should recover.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skillevo.analysis import Run, summarize  # noqa: E402
from skillevo.data import Task, save_tasks  # noqa: E402
from skillevo.evolver import validate_op  # noqa: E402
from skillevo.llm import extract_json, extract_sql  # noqa: E402
from skillevo.monitor import classify, mcnemar  # noqa: E402
from skillevo.retrieval import retrieve  # noqa: E402
from skillevo.runner import Experiment  # noqa: E402
from skillevo.skills import SkillLibrary  # noqa: E402

DEPTS = ["sales", "hr", "ops", "it", "legal", "finance", "admin", "design", "qa", "support"]


class FakeSolverLLM:
    calls = cache_hits = 0

    def chat(self, messages):
        self.calls += 1
        user = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        prompt = messages[1]["content"]
        q = prompt.split("Question:")[-1]
        dept = q.split("dept ")[-1].strip(" ?")
        if "ratio" in q:
            if "CAST" in prompt:
                return f"```sql\nSELECT CAST(SUM(active=1) AS REAL)/SUM(active=0) FROM emp WHERE dept='{dept}'\n```"
            return f"```sql\nSELECT SUM(active=1)/SUM(active=0) FROM emp WHERE dept='{dept}'\n```"
        if "COUNT(DISTINCT" in prompt:
            return f"```sql\nSELECT COUNT(DISTINCT dept) FROM emp WHERE dept='{dept}'\n```"
        return f"```sql\nSELECT COUNT(*) FROM emp WHERE dept='{dept}'\n```" + user[:0]


class FakeEvolverLLM:
    calls = cache_hits = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps([{"op": "create", "name": "Ratio with CAST", "tags": ["ratio", "divide"],
                                "description": "ratio or percentage questions",
                                "content": "Use CAST(x AS REAL) before dividing to avoid integer division."}])
        if self.calls == 2:
            return "Here you go:\n```json\n" + json.dumps([{"op": "create", "name": "Counting employees", "tags": ["employees", "count"],
                                                          "description": "count employees",
                                                          "content": "Always use COUNT(DISTINCT dept) to count."}]) + "\n```"
        return "[]"


@pytest.fixture()
def toy_root(tmp_path):
    db = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE emp (id INTEGER PRIMARY KEY, dept TEXT, salary INT, active INT)")
    rows, i = [], 0
    for d in DEPTS:
        for k in range(7):  # 5 active, 2 inactive -> ratio 2.5, integer division gives 2
            i += 1
            rows.append((i, d, 1000 + k, 1 if k < 5 else 0))
    conn.executemany("INSERT INTO emp VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    tasks = []
    for d in DEPTS:
        for v in range(3):
            tasks.append(Task(f"toy-r-{d}-{v}", "toy", "toy", str(db), f"What is the ratio of active to inactive employees{' '*v} in dept {d}?",
                              "", f"SELECT CAST(SUM(active=1) AS REAL)/SUM(active=0) FROM emp WHERE dept='{d}'", "moderate"))
            tasks.append(Task(f"toy-c-{d}-{v}", "toy", "toy", str(db), f"How many employees{' '*v} work in dept {d}?",
                              "", f"SELECT COUNT(*) FROM emp WHERE dept='{d}'", "simple"))
    save_tasks(tasks, tmp_path / "toy.jsonl")
    ids = [t.id for t in tasks]
    splits = {"toy": {"train": ids[0::3], "val": ids[1::3], "test": ids[2::3]}}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    return tmp_path


def cfg_for(root):
    return {
        "root": str(root), "seed": 0, "rounds": 3, "batch_size": 10, "workers": 1, "runs_dir": "runs",
        "data": {"task_files": ["toy.jsonl"], "splits_file": "splits.json", "train": "toy:train",
                 "val": "toy:val", "test": {"toy_test": "toy:test"}},
        "llm": {"solver": "fake", "evolver": "fake", "cache": "cache.sqlite"},
        "solver": {"top_k": 3, "min_score": 0.1, "repair": True},
        "evolver": {"max_ops": 3},
        "monitor": {"probe_relevant": 30, "probe_random": 10, "min_net": 2, "min_uses": 3, "loo": True,
                    "loo_max_tasks": 4, "rollback_min_net": 3, "audit_split": "toy_test"},
    }


def run_arm(root, arm):
    exp = Experiment(cfg_for(root), arm, solver_llm=FakeSolverLLM(), evolver_llm=FakeEvolverLLM(), log=lambda m: None)
    return Run(exp.run())


# --------------------------------------------------------------------- unit
def test_mcnemar_and_classify():
    st = mcnemar([False] * 10 + [True] * 10, [True] * 10 + [True] * 10)
    assert st["gained"] == 10 and st["lost"] == 0 and st["p_value"] < 0.01
    assert classify(st, 10, 2, 0.05).label == "positive"
    st = mcnemar([True, True, False], [False, True, False])
    assert classify(st, 3, 2, 0.05).label == "neutral"
    assert classify(mcnemar([True], [True]), 0, 2, 0.05).label == "inert"


def test_library_versions_and_merge():
    lib = SkillLibrary()
    a = lib.create("A", "dates", "use strftime")
    b = lib.create("B", "date filters", "compare as text")
    a2 = lib.update(a.id, content="use strftime('%Y', col)")
    assert a2.version == 2 and a2.parents == [f"{a.id}@v1"]
    m = lib.merge([a.id, b.id], "Dates", "dates", "merged")
    assert set(m.parents) == {f"{a.id}@v2", f"{b.id}@v1"} and len(lib) == 1
    assert SkillLibrary.from_json(lib.to_json()).get(m.id).content == "merged"


def test_retrieve_and_parsers():
    lib = SkillLibrary()
    lib.create("Ratio", "ratio percentage divide", "CAST", ["ratio"])
    lib.create("Dates", "year month date", "strftime", ["date"])
    assert [s.name for s in retrieve(lib.active(), "what is the ratio of x", 3, 0.1)] == ["Ratio"]
    assert extract_sql("blah ```sql\nSELECT 1;\n``` x") == "SELECT 1"
    assert extract_json("ok ```json\n[{\"op\":\"create\"}]\n```") == [{"op": "create"}]
    assert validate_op({"op": "update", "skill_id": "S999", "content": "x"}, lib, {"update"}) is None
    assert validate_op({"op": "merge", "skill_ids": ["S001", "S001"], "content": "x"}, lib, {"merge"}) is None


# -------------------------------------------------------------- integration
def test_monitor_separates_helpful_from_harmful(toy_root):
    g = run_arm(toy_root, "guarded")
    e = g.edits.set_index("summary")
    helpful, harmful = e.loc["create 'Ratio with CAST'"], e.loc["create 'Counting employees'"]
    assert helpful.v_label == "positive" and helpful.accepted
    assert harmful.v_label == "negative" and not harmful.accepted
    assert harmful.a_label == "negative"  # held-out audit agrees
    acc = g.accuracy()
    test = acc[acc.split == "toy_test"].sort_values("round").accuracy.tolist()
    assert test[0] == pytest.approx(0.5) and test[-1] == pytest.approx(1.0)


def test_unguarded_keeps_harmful_edit_and_drifts_down(toy_root):
    u = run_arm(toy_root, "unguarded")
    harmful = u.edits.set_index("summary").loc["create 'Counting employees'"]
    assert harmful.v_label == "negative" and harmful.accepted
    s = summarize(u)
    assert s["kept_negative"] == 1 and s["val_max_drawdown"] > 0
    stats = u.stats[u.stats["round"] == u.stats["round"].max()].set_index("name")
    # Raw lift is confounded (the harmful skill is co-retrieved with the helpful one on ratio
    # questions), leave-one-out is not: it must expose the harmful skill.
    assert stats.loc["Ratio with CAST", "lift"] > 0 and stats.loc["Ratio with CAST", "health"] == "helpful"
    assert stats.loc["Counting employees", "loo"] < 0 and stats.loc["Counting employees", "health"] == "harmful"
    assert set(u.versions.event) >= {"create"}


def test_append_only_and_noskill(toy_root):
    n = run_arm(toy_root, "noskill")
    assert len(n.rounds) == 1
    a = run_arm(toy_root, "append_only")
    assert set(a.edits.op) == {"create"}


def test_gate_only_accepts_strict_improvements_only(toy_root):
    g = run_arm(toy_root, "gate_only")
    e = g.edits.set_index("summary")
    assert e.loc["create 'Ratio with CAST'"].accepted and not e.loc["create 'Counting employees'"].accepted
