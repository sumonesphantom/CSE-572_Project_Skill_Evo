"""Offline tests for the phibench harness: toy SQLite database + a scripted fake model."""
import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phibench import compare  # noqa: E402
from phibench.examples import Demo, Pool, build_pools  # noqa: E402
from phibench.llm import Cache, ChatClient  # noqa: E402
from phibench.metrics import error_type, mcnemar, rves_reward, soft_f1  # noqa: E402
from phibench.parsing import extract_sql  # noqa: E402
from phibench.run import Bench, eval_ids, summarize  # noqa: E402
from phibench.schema import mschema  # noqa: E402
from phibench.strategies import STRATEGIES, build_messages  # noqa: E402
from skillevo.data import Task  # noqa: E402
from skillevo.solver import build_prompt as repo_prompt  # noqa: E402
from skillevo.sqlexec import schema_text  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def db(tmp_path):
    p = tmp_path / "toy.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE emp (id INTEGER PRIMARY KEY, dept_id INTEGER REFERENCES dept(id), salary INT, "Hire Year" INT);
        INSERT INTO dept VALUES (1,'sales'),(2,'hr');
        INSERT INTO emp VALUES (1,1,100,2020),(2,1,200,2021),(3,2,300,2020);
    """)
    c.commit()
    c.close()
    return str(p)


def task(db, i, q, gold, ds="toy", diff="simple", ev=""):
    return Task(f"{ds}-{i}", ds, "toy", db, q, ev, gold, diff)


class FakeLLM:
    """Wrong on the first try for 'salary' questions (bad column), right after repair."""
    model = label = "fake"
    backend, num_ctx = "fake", 0

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature=0.0, sample=0):
        self.calls += 1
        q = messages[1]["content"].split("Question:")[-1]
        repair = messages[-1]["content"].startswith("That query failed")
        if "salary" in q:
            sql = "SELECT SUM(salary) FROM emp" if repair else "SELECT SUM(pay) FROM emp"
        elif "departments" in q:
            sql = "SELECT COUNT(*) FROM dept"
        else:
            sql = "SELECT 42"
        return f"Here you go:\n```sql\n{sql};\n```\nThis sums it.", {"seconds": 0.01, "in_tokens": 10, "out_tokens": 5}


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("reply,want", [
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
    ("text ```sql\nSELECT 1\n``` more ```sql\nSELECT 2\n```", "SELECT 2"),
    ("```\nSELECT a FROM t\n```", "SELECT a FROM t"),
    ("```sql\nSELECT a FROM t WHERE b = 'x;y'", "SELECT a FROM t WHERE b = 'x;y'"),   # unterminated fence
    ("The query is SELECT a FROM t; it returns a.", "SELECT a FROM t"),
    ("SELECT a\nFROM t\n\nThis query selects a.", "SELECT a\nFROM t"),
    ("<think>hmm</think>```sql\nWITH x AS (SELECT 1) SELECT * FROM x\n```", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("```sql\nSELECT 1; SELECT 2;\n```", "SELECT 1"),
])
def test_extract_sql(reply, want):
    assert extract_sql(reply) == want


# ------------------------------------------------------------------ schema & prompts
def test_mschema_lists_types_examples_and_fks(db):
    s = mschema(db)
    assert "Table emp (" in s and "dept_id INTEGER" in s and "`Hire Year` INT" in s
    assert "e.g. 'sales', 'hr'" in s
    assert "emp.dept_id = dept.id" in s


def test_a1_is_byte_identical_to_repo_prompt(db):
    t = task(db, 1, "How many departments?", "SELECT COUNT(*) FROM dept")
    assert build_messages(t, STRATEGIES["a1_repo"]) == repo_prompt(t, [], schema_text(db, 2))


def test_rules_and_demos_in_prompt(db):
    t = task(db, 1, "How many departments?", "SELECT COUNT(*) FROM dept", ev="dept = department")
    d = Demo("x", "toy", "toy", "Count employees", "", "SELECT COUNT(*) FROM emp")
    user = build_messages(t, STRATEGIES["a4_fs3"], [d])[1]["content"]
    assert user.index("Example 1") < user.index("Guidelines:") < user.index("External knowledge: dept") < user.rindex("Question:")


# ------------------------------------------------------------------ examples
def test_pool_prefers_same_db_and_excludes_eval(db):
    t = task(db, 9, "average salary per department", "SELECT 1")
    demos = [Demo("a", "toy", "other", "average salary per department", "", "S1"),
             Demo("b", "toy", "toy", "salary of each employee", "", "S2"),
             Demo("c", "toy", "toy", "budget of a department", "", "S3")]
    assert [d.id for d in Pool(demos).top(t, 2)] == ["b", "c"]
    assert [d.id for d in Pool(demos).top(t, 1, "bm25")] == ["a"]
    tasks = {"bird-1": task(db, 1, "q1", "S", "bird"), "bird-2": task(db, 2, "q2", "S", "bird")}
    pools = build_pools(tasks, {"bird": {"train": ["bird-1", "bird-2"]}}, None, eval_ids={"bird-2"})
    assert [d.id for d in pools["bird"].demos] == ["bird-1"] and pools["bird"].dropped == ["bird-2"]


# ------------------------------------------------------------------ metrics
def test_metrics():
    assert soft_f1([(1, 2)], [(1, 2)]) == 1.0 and soft_f1([], [(1,)]) == 0.0 and soft_f1([], []) == 1.0
    assert 0 < soft_f1([(1, 9)], [(1, 2)]) < 1
    assert error_type("no such column: pay", False) == "bad_column"
    assert error_type(None, False, 0) == "empty_result" and error_type(None, True) == "correct"
    st = mcnemar([False] * 8 + [True] * 2, [True] * 8 + [False] * 2)
    assert (st["gained"], st["lost"], st["net"]) == (8, 2, 6) and 0 < st["p_value"] < 0.2


def test_rves_reward(db):
    assert rves_reward(db, "SELECT 1", "SELECT 1", False) == 0.0
    assert rves_reward(db, "SELECT COUNT(*) FROM dept", "SELECT COUNT(*) FROM dept", True) in (0.75, 1.0, 1.25)


# ------------------------------------------------------------------ splits
def test_spider_sample_matches_repo_rule():
    splits = {"spider": {"test": [f"spider-{i}" for i in range(518)]}}
    want = sorted(random.Random(0).sample(splits["spider"]["test"], 200))
    assert eval_ids(splits, "spider_test") == want


# ------------------------------------------------------------------ end to end
def _bench(db, tmp_path, llm=None):
    tasks = {t.id: t for t in [task(db, 1, "How many departments?", "SELECT COUNT(*) FROM dept", "bird"),
                               task(db, 2, "Total salary?", "SELECT SUM(salary) FROM emp", "bird", "moderate"),
                               task(db, 3, "Something odd?", "SELECT 7", "bird")]}
    splits = {"bird": {"train": [], "val": ["bird-1", "bird-2", "bird-3"], "test": ["bird-1", "bird-1"]}}
    return Bench(llm or FakeLLM(), "a2_mschema", tasks, splits, workers=1), tmp_path / "run"


def test_run_end_to_end_with_repair_and_resume(db, tmp_path):
    llm = FakeLLM()
    b, out = _bench(db, tmp_path, llm)
    m = b.run(["bird_val"], out, log=lambda _: None)["bird_val"]
    assert m["n"] == 3 and m["ex"] == pytest.approx(2 / 3) and m["ex_first"] == pytest.approx(1 / 3)
    assert m["repair_rate"] == pytest.approx(1 / 3) and m["errors"] == {"correct": 2, "wrong_result": 1}
    assert "rves" in m and m["ex_moderate"] == 1.0
    calls = llm.calls
    b.run(["bird_val"], out, log=lambda _: None)  # resume: nothing solved again
    assert llm.calls == calls
    meta = json.loads((out / "meta.json").read_text())
    assert meta["strategy"]["name"] == "a2_mschema" and meta["allow_test_used"] is False


def test_test_sets_are_sealed_and_duplicates_counted_like_repo(db, tmp_path):
    b, out = _bench(db, tmp_path)
    with pytest.raises(PermissionError):
        b.run(["bird_test"], out, log=lambda _: None)
    m = b.run(["bird_test"], out, allow_test=True, log=lambda _: None)["bird_test"]
    assert m["n"] == 2  # the repo split lists bird-137 twice; we keep that so n matches Venkat's 155


def test_meta_mismatch_refused(db, tmp_path):
    b, out = _bench(db, tmp_path)
    b.run(["bird_val"], out, log=lambda _: None)
    b2 = Bench(FakeLLM(), "a1_repo", b.tasks, b.splits, workers=1)
    with pytest.raises(ValueError):
        b2.run(["bird_val"], out, log=lambda _: None)


def test_compare_pair_and_table(db, tmp_path):
    runs = tmp_path / "runs"
    b, _ = _bench(db, tmp_path)
    b.run(["bird_val"], runs / "x__a2_mschema", log=lambda _: None)
    b2 = Bench(FakeLLM(), "a3_rules", b.tasks, b.splits, workers=1)
    b2.run(["bird_val"], runs / "x__a3_rules", log=lambda _: None)
    df = compare.load_runs(runs)
    assert set(df.strategy) == {"a2_mschema", "a3_rules"}
    assert compare.pair("x__a2_mschema", "x__a3_rules", runs)[0]["n"] == 3


# ------------------------------------------------------------------ client cache
def test_cache_key_separates_settings(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    a = ChatClient("phi4-mini", cache=c, num_ctx=16384)
    b = ChatClient("phi4-mini", cache=c, num_ctx=8192)
    lora = ChatClient("base", "openai", label="phi4mini-r16", cache=c)
    msgs = [{"role": "user", "content": "x"}]
    keys = {a.key(msgs, 0.0, 0), b.key(msgs, 0.0, 0), a.key(msgs, 0.7, 1), lora.key(msgs, 0.0, 0)}
    assert len(keys) == 4
    c.put(a.key(msgs, 0.0, 0), "SELECT 1", {"seconds": 1})
    assert a.chat(msgs) == ("SELECT 1", {"seconds": 1, "cached": True}) and a.cache_hits == 1


def test_summarize_empty():
    assert summarize([]) == {}
