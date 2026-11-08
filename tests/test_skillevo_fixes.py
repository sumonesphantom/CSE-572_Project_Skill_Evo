"""Phase C fixes for SkillEvo: C1 rejection memory, C2 database-scoped skills, optional think flag."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skillevo import llm as llm_mod  # noqa: E402
from skillevo.data import Task  # noqa: E402
from skillevo.evolver import Evolver  # noqa: E402
from skillevo.runner import Experiment  # noqa: E402
from skillevo.skills import SkillLibrary  # noqa: E402
from skillevo.solver import Solver, skill_db  # noqa: E402
from test_skillevo import FakeSolverLLM, cfg_for, toy_root  # noqa: E402,F401  (toy_root is a fixture)


def T(i, db, q="How many cards are legal?"):
    return Task(f"t{i}", "bird", db, "", q, "", "SELECT 1")


def test_db_scoped_selection():
    lib = SkillLibrary()
    g = lib.create("Legal status", "legal status filter", "use status = 'Legal'", ["legal"])
    c = lib.create("Cards via sets", "cards set join", "JOIN sets ON cards.setCode = sets.code", ["db:card_games"])
    f = lib.create("F1 races", "race results", "JOIN races", ["db:formula_1"])
    assert skill_db(c) == "card_games" and skill_db(g) is None
    on = Solver(None, top_k=3, min_score=0.1, db_skills=True)
    off = Solver(None, top_k=3, min_score=0.1)
    picked = [s.id for s in on.select_skills(T(1, "card_games"), lib)]
    assert picked[0] == c.id and g.id in picked and f.id not in picked     # own db skill always eligible
    assert f.id not in [s.id for s in on.select_skills(T(2, "card_games", "races?"), lib)]
    # default (fix off) = repo behaviour: plain BM25 over all skills, no database scoping
    assert {s.id for s in off.select_skills(T(3, "card_games"), lib)} == {g.id, c.id}
    assert c.id in [s.id for s in off.select_skills(T(4, "formula_1", "How many cards are legal?"), lib)]
    assert c.id not in [s.id for s in on.select_skills(T(4, "formula_1", "How many cards are legal?"), lib)]


class CaptureEvolver:
    def __init__(self, reply="[]"):
        self.reply, self.msgs = reply, None
        self.calls = self.cache_hits = 0

    def chat(self, messages):
        self.msgs = messages
        return self.reply


def test_evolver_prompt_rejections_and_db():
    tasks = {"t1": T(1, "card_games")}
    trajs = [{"task_id": "t1", "skills": [], "sql": "SELECT 0", "correct": False, "error": None}]
    rej = [{"summary": "create 'Percentages'", "label": "neutral", "gained": 1, "lost": 2, "content": "CAST AS REAL"}]
    llm = CaptureEvolver(json.dumps([{"op": "create", "name": "x", "content": "y", "db": "card_games"},
                                     {"op": "create", "name": "z", "content": "w", "db": "no_such_db"}]))
    ops, _ = Evolver(llm, db_skills=True, show_rejections=True).propose(SkillLibrary(), trajs, tasks, rejected=rej)
    user, system = llm.msgs[1]["content"], llm.msgs[0]["content"]
    assert "PREVIOUSLY REJECTED" in user and "fixed 1, broke 2" in user and "Database: card_games" in user
    assert '"db"' in system
    assert ops[0]["tags"] == ["db:card_games"] and ops[1]["tags"] == [] and "db" not in ops[0]
    llm2 = CaptureEvolver(json.dumps([{"op": "create", "name": "x", "content": "y", "tags": ["db:card_games"]}]))
    ops2, _ = Evolver(llm2).propose(SkillLibrary(), trajs, tasks, rejected=rej)       # fixes off
    assert "PREVIOUSLY REJECTED" not in llm2.msgs[1]["content"] and "Database:" not in llm2.msgs[1]["content"]
    assert ops2[0]["tags"] == []                                                     # no db skills without the fix


class EvolverWithCapture:
    """Round 1: helpful ratio skill; round 2: harmful count skill (rejected); records every prompt."""
    def __init__(self):
        self.calls = self.cache_hits = 0
        self.prompts = []

    def chat(self, messages):
        self.calls += 1
        self.prompts.append(messages[1]["content"])
        if self.calls == 1:
            return json.dumps([{"op": "create", "name": "Ratio with CAST", "tags": ["ratio", "divide"],
                                "description": "ratio or percentage questions",
                                "content": "Use CAST(x AS REAL) before dividing to avoid integer division."}])
        if self.calls == 2:
            return json.dumps([{"op": "create", "name": "Counting employees", "tags": ["employees", "count"],
                                "description": "count employees", "content": "Always use COUNT(DISTINCT dept) to count."}])
        return "[]"


def test_runner_feeds_rejections_back(toy_root):  # noqa: F811
    cfg = cfg_for(toy_root)
    cfg["evolver"]["show_rejections"] = True
    ev = EvolverWithCapture()
    Experiment(cfg, "guarded", "fix", solver_llm=FakeSolverLLM(), evolver_llm=ev, log=lambda m: None).run()
    assert "PREVIOUSLY REJECTED" not in ev.prompts[1]
    assert "PREVIOUSLY REJECTED" in ev.prompts[2] and "Counting employees" in ev.prompts[2]


def test_think_flag_optional(monkeypatch):
    sent = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    monkeypatch.setattr(llm_mod.requests, "post", lambda url, json, timeout: sent.append(json) or R())
    llm_mod.OllamaLLM("m").chat([{"role": "user", "content": "a"}])
    llm_mod.OllamaLLM("m", think=None).chat([{"role": "user", "content": "b"}])
    assert sent[0]["think"] is False and "think" not in sent[1]
