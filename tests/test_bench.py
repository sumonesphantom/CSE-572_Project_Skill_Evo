"""Offline tests for the benchmark harness (no Ollama): metrics, prompts, voting, few-shot retrieval."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import run as brun  # noqa: E402
from bench.fewshot import Demo, DemoPool  # noqa: E402
from bench.metrics import error_type, rves_reward, soft_f1  # noqa: E402
from bench.schema import mschema  # noqa: E402
from bench.strategies import LADDER, build_messages  # noqa: E402
from skillevo.data import Task  # noqa: E402
from skillevo.llm import make_llm  # noqa: E402
from skillevo.solver import build_prompt  # noqa: E402
from skillevo.sqlexec import schema_text  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "toy.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE emp (id INTEGER PRIMARY KEY, "full name" TEXT, dept_id INTEGER REFERENCES dept(id), salary REAL);
        INSERT INTO dept VALUES (1,'sales'),(2,'hr');
        INSERT INTO emp VALUES (1,'Ann',1,10),(2,'Bob',1,20),(3,'Cy',2,30);
    """)
    c.commit()
    c.close()
    return str(p)


def task(db, q="How many employees are in sales?", gold="SELECT COUNT(*) FROM emp WHERE dept_id=1", tid="t1"):
    return Task(tid, "bird", "toy", db, q, "", gold, "simple")


def test_soft_f1_matches_bird_definition():
    assert soft_f1([(1, "a")], [(1, "a")]) == 1.0
    assert soft_f1([], []) == 1.0
    assert soft_f1([], [(1,)]) == 0.0
    # one of two cells right in a single row -> precision 0.5, recall 0.5
    assert soft_f1([(1, "x")], [(1, "a")]) == pytest.approx(0.5)
    # extra predicted row lowers precision only
    assert 0 < soft_f1([(1,), (2,)], [(1,)]) < 1


def test_error_types():
    assert error_type(None, True) == "correct"
    assert error_type(None, False) == "wrong_result"
    assert error_type("no such column: foo", False) == "bad_column"
    assert error_type('near "SELEC": syntax error', False) == "syntax"


def test_rves_zero_when_wrong(db):
    assert rves_reward(db, "SELECT 1", "SELECT 2", correct=False) == 0.0
    assert rves_reward(db, "SELECT COUNT(*) FROM emp", "SELECT COUNT(*) FROM emp", correct=True) in (0.25, 0.5, 0.75, 1.0, 1.25)


def test_zs_repo_prompt_is_identical_to_repo_solver(db):
    t = task(db)
    assert build_messages(t, LADDER["zs_repo"]) == build_prompt(t, [], schema_text(db, 2))


def test_mschema_has_types_examples_fk_and_quotes(db):
    s = mschema(db)
    assert '"full name": TEXT' in s and "e.g. 'Ann'" in s and "PK" in s
    assert "emp.dept_id = dept.id" in s


def test_rules_and_cot_prompts(db):
    t = task(db)
    m = build_messages(t, LADDER["fs3_cot"], demos=[Demo("d1", "toy", "Count hr staff", "", "SELECT COUNT(*) FROM emp WHERE dept_id=2")])
    assert "think briefly" in m[0]["content"]
    assert "Rules:" in m[1]["content"] and "Solved examples" in m[1]["content"]
    assert m[1]["content"].rstrip().endswith("Question: How many employees are in sales?")


def test_demo_pool_prefers_same_db_and_skips_self():
    pool = DemoPool([Demo("t1", "toy", "How many employees in sales", "", "x"),
                     Demo("a", "other", "How many employees in sales department", "", "y"),
                     Demo("b", "toy", "List salaries", "", "z")])
    t = Task("t1", "bird", "toy", "", "How many employees are in sales?", "", "")
    top = pool.top(t, 2)
    assert all(d.id != "t1" for d in top)          # never leak the task itself
    assert {d.id for d in top} == {"a", "b"}


class ScriptedLLM:
    """Greedy answer is wrong; 3 of 4 samples agree on the right answer -> vote must pick it."""
    model = "fake"
    calls = cache_hits = 0

    def chat_meta(self, msgs, temperature=0.0, sample=0):
        right, wrong = "SELECT COUNT(*) FROM emp WHERE dept_id=1", "SELECT COUNT(*) FROM emp"
        sql = wrong if sample in (0, 4) else right
        return f"```sql\n{sql}\n```", {"cached": False, "seconds": 0.0, "in_tokens": 10, "out_tokens": 5}


def _bench(strategy):
    b = brun.Bench.__new__(brun.Bench)
    b.strat, b.llm, b.library, b.repair, b.ves, b.pools = LADDER[strategy], ScriptedLLM(), None, True, False, {}
    b.top_k, b.min_score = 3, 0.5
    return b


def test_self_consistency_vote(db):
    t = task(db)
    greedy = _bench("zs_rules").solve(t)
    assert not greedy["correct"]
    voted = _with_no_shots(_bench("fs3_sc5")).solve(t)
    assert voted["correct"] and voted["vote_size"] == 3 and voted["oracle_correct"] and voted["n_candidates"] == 5


def _with_no_shots(b):
    from dataclasses import replace
    b.strat = replace(b.strat, k_shots=0)
    return b


def test_repair_turn_and_first_attempt(db):
    class Broken(ScriptedLLM):
        def chat_meta(self, msgs, temperature=0.0, sample=0):
            sql = "SELECT COUNT(*) FROM emp WHERE dept_id=1" if len(msgs) > 2 else "SELECT nope FROM emp"
            return f"```sql\n{sql}\n```", {"cached": True}
    b = _bench("zs_rules")
    b.llm = Broken()
    r = b.solve(task(db))
    assert r["correct"] and r["repaired"] and not r["first_correct"] and r["llm_calls"] == 2


def test_make_llm_routes_mlx():
    llm = make_llm("mlx:coder3b-r16@localhost:9000")
    assert llm.host == "http://localhost:9000" and llm.api_model == "default_model" and llm.model == "mlx:coder3b-r16"
    assert brun.run_name("mlx:coder3b-r16@localhost:9000", "ms_rules") == "mlx-coder3b-r16__ms_rules"
    assert make_llm("qwen2.5-coder:7b").api_model == "qwen2.5-coder:7b"


def test_greedy_cache_key_unchanged():
    llm = make_llm("m")
    assert llm._key([{"role": "user", "content": "x"}]) == llm._key([{"role": "user", "content": "x"}], 0.0, 0)
    assert llm._key([{"role": "user", "content": "x"}]) != llm._key([{"role": "user", "content": "x"}], 0.7, 1)
