"""Self-evolution loop and experiment arms.

Arms
- noskill     : the base agent, no library (lower baseline).
- append_only : evolver may only CREATE; every proposal is kept (Voyager/ExpeL-style accumulation).
- unguarded   : full create/update/merge/discard; every proposal is kept. The monitor still
                scores every edit, so negative drift is *observed* but not prevented.
- gate_only   : full lifecycle; an edit is kept only if it strictly improves accuracy on the
                whole validation set (the SkillOpt / "Rethinking" style aggregate gate). No
                per-skill attribution, no stats shown to the evolver, no rollback.
- guarded     : full lifecycle; an edit is kept only if the monitor accepts it, the evolver
                sees per-skill lift, harmful skills are auto-discarded and the library is
                rolled back to the best snapshot on a significant validation drop.

Everything is logged as JSON lines under runs/<run_name>/ for the dashboard.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import load_tasks
from .evolver import Evolver, apply_op, describe_op
from .llm import LLMCache, OllamaLLM
from .monitor import Monitor
from .skills import SkillLibrary, text_diff
from .solver import Solver

ARMS = ("noskill", "append_only", "unguarded", "gate_only", "guarded")


class Logger:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, stream: str, rec: dict):
        rec = {"ts": time.time(), **rec}
        with open(self.dir / f"{stream}.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


class Experiment:
    def __init__(self, cfg: dict, arm: str, run_name: str | None = None,
                 solver_llm=None, evolver_llm=None, log=print):
        assert arm in ARMS, f"arm must be one of {ARMS}"
        self.cfg, self.arm, self.say = cfg, arm, log
        self.seed = cfg.get("seed", 0)
        root = Path(cfg.get("root", "."))
        self.run_dir = root / cfg.get("runs_dir", "runs") / (run_name or f"{arm}-s{self.seed}")
        if self.run_dir.exists() and any(self.run_dir.glob("*.jsonl")):
            raise FileExistsError(f"{self.run_dir} already has logs; pick another --name or delete it")
        self.log = Logger(self.run_dir)

        self.tasks = {}
        for path in cfg["data"]["task_files"]:
            self.tasks.update(load_tasks(root / path))
        splits = json.loads((root / cfg["data"]["splits_file"]).read_text())
        self.train_ids = self._split(splits, cfg["data"]["train"])
        self.val_ids = self._split(splits, cfg["data"]["val"])
        self.eval_sets = {name: self._split(splits, spec) for name, spec in cfg["data"]["test"].items()}

        cache = LLMCache(root / cfg["llm"]["cache"])
        host = cfg["llm"].get("host", "http://localhost:11434")
        self.solver_llm = solver_llm or OllamaLLM(cfg["llm"]["solver"], host, cache, num_ctx=cfg["llm"].get("solver_ctx", 8192), max_tokens=512, think=cfg["llm"].get("think", False))
        self.evolver_llm = evolver_llm or OllamaLLM(cfg["llm"]["evolver"], host, cache, num_ctx=cfg["llm"].get("evolver_ctx", 16384), max_tokens=2048, think=cfg["llm"].get("think", False))
        sc = cfg["solver"]
        self.solver = Solver(self.solver_llm, sc.get("top_k", 3), sc.get("min_score", 0.5), sc.get("repair", True), sc.get("sql_timeout", 30),
                             db_skills=sc.get("db_skills", False), max_db_skills=sc.get("max_db_skills", 2))
        ec = cfg["evolver"]
        allowed = ("create",) if arm == "append_only" else ("create", "update", "merge", "discard")
        self.evolver = Evolver(self.evolver_llm, ec.get("max_ops", 4), ec.get("n_fail", 10), ec.get("n_success", 3),
                               ec.get("show_gold", True), show_stats=(arm == "guarded"), allowed_ops=allowed, seed=self.seed,
                               show_rejections=ec.get("show_rejections", False), db_skills=ec.get("db_skills", False),
                               enforce=ec.get("enforce", False))
        self.monitor = Monitor(self.solver, self.tasks, self.val_ids, cfg["monitor"], seed=self.seed)
        # Audit monitor: re-tests each edit on held-out test tasks the gate never sees, so we can
        # measure whether validation verdicts generalise (verdict agreement / precision).
        if arm == "gate_only":  # aggregate gate: probe = the entire validation set
            self.gate = Monitor(self.solver, self.tasks, self.val_ids,
                                {**cfg["monitor"], "probe_relevant": 10**6, "probe_random": 10**6}, seed=self.seed)
        audit = cfg["monitor"].get("audit_split")
        self.auditor = Monitor(self.solver, self.tasks, self.eval_sets[audit], cfg["monitor"], seed=self.seed + 1) if audit else None
        self.workers = cfg.get("workers", 2)
        self.library = SkillLibrary()
        self.rejected: list[dict] = []   # edits the arm did not keep (shown to the evolver when show_rejections)

    def _split(self, splits, spec):
        """spec is 'bird:train' or {'split': 'spider:test', 'max': 200}."""
        if isinstance(spec, str):
            spec = {"split": spec}
        ds, name = spec["split"].split(":")
        ids = list(splits[ds][name])
        if spec.get("max") and len(ids) > spec["max"]:
            ids = sorted(random.Random(self.seed).sample(ids, spec["max"]))
        return ids

    # ------------------------------------------------------------------ helpers
    def solve_many(self, ids, library, phase, round_):
        with ThreadPoolExecutor(self.workers) as ex:
            trajs = list(ex.map(lambda tid: self.solver.solve(self.tasks[tid], library), ids))
        out = []
        for tr in trajs:
            d = tr.to_dict()
            d.update(round=round_, phase=phase, baseline=self.monitor.baseline(tr.task_id),
                     difficulty=self.tasks[tr.task_id].difficulty, dataset=self.tasks[tr.task_id].dataset)
            self.log.write("trajectories", d)
            out.append(d)
        return out

    def record_version(self, skill, round_, event, edit_id=None, parent_text=None):
        self.log.write("skill_versions", {
            "round": round_, "event": event, "edit_id": edit_id, "skill_id": skill.id, "version": skill.version,
            "key": skill.key, "name": skill.name, "description": skill.description, "content": skill.content,
            "tags": skill.tags, "parents": skill.parents, "origin": skill.origin,
            "diff": text_diff(parent_text, skill.text()) if parent_text else None,
        })

    def evaluate(self, round_, which="all"):
        res = {}
        val = self.solve_many(self.val_ids, self.library, "val", round_)
        res["val"] = val
        final = which == "final"
        every = self.cfg.get("eval_every", 1)
        for name, ids in self.eval_sets.items():
            if final or round_ == 0 or round_ % every == 0:
                res[name] = self.solve_many(ids, self.library, name, round_)
        return res

    # ------------------------------------------------------------------ main loop
    def run(self, rounds: int | None = None):
        rounds = self.cfg.get("rounds", 8) if rounds is None else rounds
        if self.arm == "noskill":
            rounds = 0
        (self.run_dir / "config.json").write_text(json.dumps({"arm": self.arm, **self.cfg}, indent=2, default=str))
        self.say(f"[{self.arm}] tasks: train={len(self.train_ids)} val={len(self.val_ids)} "
                 + " ".join(f"{k}={len(v)}" for k, v in self.eval_sets.items()))

        evals = self.evaluate(0)
        best_val = prev_val = {t["task_id"]: t["correct"] for t in evals["val"]}
        best_lib, best_round = self.library.clone(), 0
        stats = {}
        self._log_round(0, evals, {}, stats, edits=[], rolled_back=False, best_round=0)

        pool = list(self.train_ids)
        rng = random.Random(self.seed)
        rng.shuffle(pool)
        bsz = self.cfg.get("batch_size", 40)
        edit_counter = 0
        for r in range(1, rounds + 1):
            t0 = time.time()
            if len(pool) < bsz:
                extra = list(self.train_ids)
                rng.shuffle(extra)
                pool += extra
            batch, pool = pool[:bsz], pool[bsz:]
            trajs = self.solve_many(batch, self.library, "train", r)
            train_acc = sum(t["correct"] for t in trajs) / len(trajs)
            self.say(f"  round {r}: train acc {train_acc:.3f}; evolving ...")

            ops, raw = self.evolver.propose(self.library, trajs, self.tasks, stats, rejected=self.rejected[-12:])
            self.log.write("evolver", {"round": r, "n_ops": len(ops), "raw": raw,
                                       "filtered": getattr(self.evolver, "last_dropped", {})})

            if self.arm == "guarded":  # monitor-initiated discards of skills proven harmful
                for sid, st in stats.items():
                    if st["health"] == "harmful" and self.library.get(sid) and not any(o.get("skill_id") == sid for o in ops):
                        ops.append({"op": "discard", "skill_id": sid, "tags": [], "rationale": "monitor: negative lift", "source": "monitor"})

            edits = []
            for op in ops:
                if op["op"] in ("update", "discard") and self.library.get(op["skill_id"]) is None:
                    continue
                if op["op"] == "merge" and any(self.library.get(i) is None for i in op["skill_ids"]):
                    continue
                edit_counter += 1
                eid = f"E{edit_counter:04d}"
                before = {sid: self.library.get(sid) for sid in ([op["skill_id"]] if "skill_id" in op else op.get("skill_ids", []))}
                cand = self.library.clone()
                touched, new_skill = apply_op(cand, op, r)
                verdict, probe = self.monitor.evaluate_edit(self.library, cand, touched, new_skill.id if new_skill else None)
                accepted = self.monitor.accepts(op["op"], verdict) if self.arm == "guarded" else True
                if self.arm == "gate_only":
                    gate_verdict = self.gate.evaluate_edit(self.library, cand, touched, new_skill.id if new_skill else None)[0]
                    accepted = gate_verdict.net > 0
                audit = self.auditor.evaluate_edit(self.library, cand, touched, new_skill.id if new_skill else None)[0] if self.auditor else None
                rec = {"round": r, "edit_id": eid, "op": op["op"], "summary": describe_op(op),
                       "source": op.get("source", "evolver"), "rationale": op.get("rationale", ""),
                       "skills_before": [s.key for s in before.values() if s], "skill_after": new_skill.key if new_skill else None,
                       "accepted": accepted, **{f"v_{k}": v for k, v in verdict.to_dict().items()},
                       **({f"a_{k}": v for k, v in audit.to_dict().items()} if audit else {})}
                self.log.write("edits", rec)
                edits.append(rec)
                if not accepted and op.get("source", "evolver") == "evolver":
                    self.rejected.append({"summary": rec["summary"], "label": verdict.label, "gained": verdict.gained,
                                          "lost": verdict.lost, "content": op.get("content", "")})
                mark = "+" if verdict.label == "positive" else "-" if verdict.label == "negative" else "~"
                self.say(f"    {eid} {rec['summary'][:60]:<60} {mark} {verdict.label:<8} net={verdict.net:+d} "
                         f"p={verdict.p_value:.2f} {'KEPT' if accepted else 'rejected'}")
                if accepted:
                    self.library = cand
                    if new_skill:
                        parent = before.get(new_skill.id)
                        self.record_version(new_skill, r, op["op"], eid, parent.text() if parent else None)
                    for sid, s in before.items():
                        if s and self.library.get(sid) is None:
                            self.log.write("skill_versions", {"round": r, "event": "retired", "edit_id": eid, "skill_id": sid,
                                                              "version": s.version, "key": s.key, "name": s.name,
                                                              "by": op["op"], "into": new_skill.key if new_skill else None})

            evals = self.evaluate(r, "final" if r == rounds else "all")
            cur_val = {t["task_id"]: t["correct"] for t in evals["val"]}
            stats = self.monitor.skill_stats(self.library, evals["val"], loo=self.cfg["monitor"].get("loo", True))
            drift = self.monitor.round_drift(prev_val, best_val, cur_val)
            rolled_back = False
            if self.arm == "guarded" and self.monitor.should_rollback(drift):
                self.say(f"    ROLLBACK to round {best_round} snapshot (val net {drift['vs_best']['net']:+d} vs best)")
                self.log.write("events", {"round": r, "type": "rollback", "to_round": best_round, "drift": drift})
                self.library = best_lib.clone()
                rolled_back = True
                evals = self.evaluate(r, "final" if r == rounds else "all")
                cur_val = {t["task_id"]: t["correct"] for t in evals["val"]}
                stats = self.monitor.skill_stats(self.library, evals["val"], loo=False)
            if sum(cur_val.values()) > sum(best_val.values()):
                best_val, best_lib, best_round = cur_val, self.library.clone(), r
            prev_val = cur_val
            self._log_round(r, evals, drift, stats, edits, rolled_back, best_round, train_acc, time.time() - t0)
        self.say(f"[{self.arm}] done -> {self.run_dir}")
        return self.run_dir

    def _log_round(self, r, evals, drift, stats, edits, rolled_back, best_round, train_acc=None, secs=0.0):
        self.library.save(self.run_dir / "library" / f"round_{r:02d}.json")
        acc = {name: sum(t["correct"] for t in ts) / len(ts) for name, ts in evals.items() if ts}
        for sid, st in stats.items():
            self.log.write("skill_stats", {"round": r, **st})
        verdicts = [e["v_label"] for e in edits]
        rec = {"round": r, "arm": self.arm, "acc": acc, "train_acc": train_acc, "library_size": len(self.library),
               "library_fp": self.library.fingerprint(), "n_edits": len(edits),
               "n_kept": sum(e["accepted"] for e in edits),
               **{f"n_{lbl}": verdicts.count(lbl) for lbl in ("positive", "neutral", "negative", "inert")},
               "drift": drift, "rolled_back": rolled_back, "best_round": best_round, "seconds": round(secs, 1),
               "llm_calls": self.solver_llm.calls + self.evolver_llm.calls,
               "cache_hits": self.solver_llm.cache_hits + self.evolver_llm.cache_hits}
        self.log.write("rounds", rec)
        self.say(f"  round {r} | " + " ".join(f"{k}={v:.3f}" for k, v in acc.items())
                 + f" | skills={len(self.library)} | {secs:.0f}s")
