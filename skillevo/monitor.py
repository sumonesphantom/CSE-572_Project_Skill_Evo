"""Evolution-quality monitor: is the skill library getting better or worse?

Three levels of evidence, all *paired* (same tasks, same deterministic solver),
so differences come from the skills and not from sampling noise:

1. Edit level   - every proposed create/update/merge/discard is A/B tested on a
                  probe set of validation tasks (tasks that retrieve the touched
                  skills + a random slice to catch collateral damage). Outcome
                  flips are tested with an exact McNemar test and the edit is
                  labelled positive / neutral / negative / inert.
2. Skill level  - each skill's *lift*: success with the library minus success of
                  the cached no-skill baseline on the tasks where the skill was
                  retrieved (bootstrap CI), plus an optional leave-one-out
                  marginal effect (same context minus that one skill).
3. Round level  - validation accuracy per round compared with the previous
                  round and the best-so-far snapshot (drawdown); the guarded arm
                  rolls back when the drop is significant.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binomtest


# --------------------------------------------------------------------- stats
def mcnemar(old: list[bool], new: list[bool]) -> dict:
    """Paired comparison of two outcome vectors on the same tasks."""
    assert len(old) == len(new)
    gained = sum(1 for o, n in zip(old, new) if not o and n)
    lost = sum(1 for o, n in zip(old, new) if o and not n)
    disc = gained + lost
    p = binomtest(gained, disc, 0.5).pvalue if disc else 1.0
    n = len(old)
    return {"n": n, "gained": gained, "lost": lost, "net": gained - lost,
            "acc_old": sum(old) / n if n else 0.0, "acc_new": sum(new) / n if n else 0.0,
            "delta": (gained - lost) / n if n else 0.0, "p_value": float(p)}


def bootstrap_ci(x: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 0):
    if len(x) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@dataclass
class Verdict:
    label: str            # positive | neutral | negative | inert
    significant: bool
    n_probe: int
    n_relevant: int
    gained: int
    lost: int
    net: int
    delta: float
    p_value: float
    acc_old: float
    acc_new: float

    def to_dict(self):
        return asdict(self)


def classify(stats: dict, n_relevant: int, min_net: int, alpha: float) -> Verdict:
    if n_relevant == 0 and stats["gained"] == stats["lost"] == 0:
        label = "inert"
    elif stats["net"] >= min_net:
        label = "positive"
    elif stats["net"] <= -min_net:
        label = "negative"
    else:
        label = "neutral"
    return Verdict(label, stats["p_value"] < alpha, stats["n"], n_relevant, stats["gained"], stats["lost"],
                   stats["net"], stats["delta"], stats["p_value"], stats["acc_old"], stats["acc_new"])


ACCEPT = {  # guarded-arm acceptance rule per operation
    "create": {"positive"},
    "update": {"positive", "neutral"},
    "merge": {"positive", "neutral", "inert"},     # neutral merge = smaller library, same accuracy
    "discard": {"positive", "neutral", "inert"},
}


# ------------------------------------------------------------------- monitor
class Monitor:
    def __init__(self, solver, tasks: dict, val_ids: list[str], cfg: dict, seed: int = 0):
        self.solver = solver
        self.tasks = tasks
        self.val_ids = list(val_ids)
        self.n_relevant = cfg.get("probe_relevant", 30)
        self.n_random = cfg.get("probe_random", 20)
        self.min_net = cfg.get("min_net", 2)
        self.alpha = cfg.get("alpha", 0.05)
        self.min_uses = cfg.get("min_uses", 5)
        self.loo_max = cfg.get("loo_max_tasks", 8)
        self.rollback_min_net = cfg.get("rollback_min_net", 3)
        self.rng = random.Random(seed)
        self._baseline: dict[str, bool] = {}

    # ---- no-skill baseline (cached) -------------------------------------
    def baseline(self, task_id: str) -> bool:
        if task_id not in self._baseline:
            self._baseline[task_id] = self.solver.solve(self.tasks[task_id], skills=[]).correct
        return self._baseline[task_id]

    # ---- edit level ------------------------------------------------------
    def probe_set(self, old_lib, new_lib, touched_old: list[str], new_skill_id: str | None):
        relevant, rest = [], []
        for tid in self.val_ids:
            t = self.tasks[tid]
            old_ids = {s.id for s in self.solver.select_skills(t, old_lib)}
            new_ids = {s.id for s in self.solver.select_skills(t, new_lib)}
            hit = bool(old_ids & set(touched_old)) or (new_skill_id is not None and new_skill_id in new_ids)
            (relevant if hit else rest).append(tid)
        rel = self.rng.sample(relevant, min(len(relevant), self.n_relevant))
        rnd = self.rng.sample(rest, min(len(rest), self.n_random))
        return rel, rnd

    def evaluate_edit(self, old_lib, new_lib, touched_old, new_skill_id):
        rel, rnd = self.probe_set(old_lib, new_lib, touched_old, new_skill_id)
        probe = rel + rnd
        old = [self.solver.solve(self.tasks[t], old_lib).correct for t in probe]
        new = [self.solver.solve(self.tasks[t], new_lib).correct for t in probe]
        verdict = classify(mcnemar(old, new), len(rel), self.min_net, self.alpha)
        return verdict, probe

    @staticmethod
    def accepts(op_kind: str, verdict: Verdict) -> bool:
        return verdict.label in ACCEPT[op_kind]

    # ---- skill level -----------------------------------------------------
    def skill_stats(self, library, trajs: list[dict], loo: bool = True) -> dict[str, dict]:
        """Attribution from one evaluation pass (trajectories on val tasks)."""
        by_skill: dict[str, list[tuple[str, bool]]] = {s.id: [] for s in library.active()}
        for tr in trajs:
            for key in tr["skills"]:
                sid = key.split("@")[0]
                if sid in by_skill:
                    by_skill[sid].append((tr["task_id"], tr["correct"]))
        out = {}
        for sid, uses in by_skill.items():
            s = library.get(sid)
            diffs = np.array([float(c) - float(self.baseline(t)) for t, c in uses])
            lift = float(diffs.mean()) if len(diffs) else 0.0
            lo, hi = bootstrap_ci(diffs)
            rec = {"skill_id": sid, "version": s.version, "name": s.name, "n": len(uses),
                   "success": float(np.mean([c for _, c in uses])) if uses else 0.0,
                   "lift": lift, "ci_lo": lo, "ci_hi": hi, "loo": None, "loo_n": 0}
            if loo and uses and self.loo_max:
                sample = self.rng.sample(uses, min(len(uses), self.loo_max))
                marg = []
                for tid, c in sample:
                    t = self.tasks[tid]
                    ctx = [x for x in self.solver.select_skills(t, library) if x.id != sid]
                    marg.append(float(c) - float(self.solver.solve(t, skills=ctx).correct))
                rec["loo"], rec["loo_n"] = float(np.mean(marg)), len(marg)
            rec["health"] = self.health(rec)
            out[sid] = rec
        return out

    def health(self, rec) -> str:
        # Lift is correlational: a harmful skill co-retrieved with a helpful one inherits its
        # gains. Leave-one-out isolates the skill, so it takes precedence when available.
        if rec["n"] == 0:
            return "unused"
        if rec["n"] < self.min_uses:
            return "uncertain"
        loo = rec["loo"] if rec["loo_n"] >= min(4, rec["n"]) else None
        if rec["ci_hi"] < 0 or (loo is not None and loo < 0):
            return "harmful"
        if rec["ci_lo"] > 0 and (loo is None or loo >= 0):
            return "helpful"
        return "uncertain"

    # ---- round level -----------------------------------------------------
    def round_drift(self, prev: dict[str, bool] | None, best: dict[str, bool] | None, cur: dict[str, bool]):
        res = {}
        for name, ref in (("vs_prev", prev), ("vs_best", best)):
            if ref:
                ids = sorted(set(ref) & set(cur))
                st = mcnemar([ref[i] for i in ids], [cur[i] for i in ids])
                st["label"] = "positive" if st["net"] >= self.min_net else "negative" if st["net"] <= -self.min_net else "neutral"
                res[name] = st
        return res

    def should_rollback(self, drift: dict) -> bool:
        vb = drift.get("vs_best")
        return bool(vb and vb["net"] <= -self.rollback_min_net)
