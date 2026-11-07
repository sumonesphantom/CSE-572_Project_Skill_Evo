"""Benchmark one (model, strategy[, skill library]) on fixed BIRD / Spider eval sets.

    python -m bench.run --model qwen2.5-coder:7b --strategy zs_repo
    python -m bench.run --model qwen2.5-coder:7b --strategy fs3 --sets bird_val,bird_test,spider_test
    python -m bench.run --model qwen2.5-coder:7b --strategy fs3 --library runs/qwen7b-guarded/library/round_08.json
    python -m bench.run --model mlx:coder7b-r16 --strategy zs_rules   # needs mlx_lm.server on :8080
    python -m bench.run --model qwen2.5-coder:7b --ladder                                 # every rung P0..P8

Outputs runs/bench/<run>/{preds.jsonl, metrics.json}; reruns resume from preds.jsonl and the
shared LLM cache (data/cache/llm_cache.sqlite), so nothing is solved twice.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillevo.data import load_tasks  # noqa: E402
from skillevo.llm import LLMCache, extract_sql, make_llm  # noqa: E402
from skillevo.retrieval import retrieve, task_query  # noqa: E402
from skillevo.skills import SkillLibrary  # noqa: E402
from skillevo.sqlexec import execute  # noqa: E402

from bench.extract import extract_sql as robust_extract_sql  # noqa: E402
from bench.fewshot import build_pools  # noqa: E402
from bench.metrics import error_type, ex_summary, gold_rows, rves_reward, soft_f1  # noqa: E402
from bench.strategies import LADDER, RUNG, build_messages  # noqa: E402

EVAL_SETS = {  # name -> (dataset:split, max) ; spider_test uses the same 200-task sample as config.yaml
    "bird_val": ("bird:val", None),
    "bird_test": ("bird:test", None),
    "spider_test": ("spider:test", 200),
    "spider_val": ("spider:val", 100),      # selection set for the small-model rungs (as in the Llama 1B benchmark)
    "bird_all": ("bird:*", None),           # all 500 mini-dev questions (leaderboard-comparable; zero-shot only)
    "spider_test_full": ("spider:test", None),
}
SPIDER_TRAIN = ROOT / "data/raw/spider/spider_data/train_spider.json"


def eval_ids(splits, name, seed=0):
    spec, mx = EVAL_SETS[name]
    ds, sp = spec.split(":")
    ids = [i for v in splits[ds].values() for i in v] if sp == "*" else list(splits[ds][sp])
    if mx and len(ids) > mx:
        ids = sorted(random.Random(seed).sample(ids, mx))
    return ids


def run_name(model, strategy, library=None, tag=None):
    model = model.split("@")[0]
    m = re.sub(r"[^A-Za-z0-9.]+", "-", model.replace("mlx:", "mlx-") if model.startswith("mlx:") else model)
    parts = [m.strip("-"), strategy]
    if library:
        lp = Path(library)
        parts.append("skills-" + (lp.parent.parent.name if lp.parent.name == "library" else lp.stem))
    if tag:
        parts.append(tag)
    return "__".join(parts)


class Bench:
    def __init__(self, model, strategy, library=None, host="http://localhost:11434", num_ctx=16384,
                 top_k=3, min_score=0.5, repair=True, ves=True, workers=2):
        self.strat = LADDER[strategy]
        self.cache = LLMCache(ROOT / "data/cache/llm_cache.sqlite")
        # max_tokens is part of the cache key: 512 matches the SkillEvo solver, so zs_repo shares its cache
        self.llm = make_llm(model, host, self.cache, num_ctx=num_ctx, max_tokens=self.strat.max_tokens)
        self.library = SkillLibrary.load(library) if library else None
        self.library_path = library
        self.top_k, self.min_score, self.repair, self.ves, self.workers = top_k, min_score, repair, ves, workers
        self.num_ctx = num_ctx
        self.extract = robust_extract_sql if self.strat.extract == "robust" else extract_sql
        self.tasks = {}
        for f in ("bird", "spider"):
            self.tasks.update(load_tasks(ROOT / f"data/processed/{f}.jsonl"))
        self.splits = json.loads((ROOT / "data/processed/splits.json").read_text())
        self.pools = build_pools(self.tasks, self.splits, SPIDER_TRAIN) if self.strat.k_shots else {}

    # ---------------------------------------------------------------- one candidate
    def _attempt(self, task, msgs, temperature, sample):
        reply, meta = self.llm.chat_meta(msgs, temperature, sample)
        sql = self.extract(reply)
        rows, err = execute(task.db_path, sql, 30)
        metas = [meta]
        first = {"sql": sql, "err": err, "rows": rows}
        if self.repair and err:
            msgs2 = msgs + [{"role": "assistant", "content": reply},
                            {"role": "user", "content": f"That query failed with: {err}\nFix it. Return only the corrected ```sql block."}]
            reply2, meta2 = self.llm.chat_meta(msgs2, temperature, sample)
            metas.append(meta2)
            sql2 = self.extract(reply2)
            rows, err = execute(task.db_path, sql2, 30)
            return {"sql": sql2, "err": err, "rows": rows, "first": first, "repaired": True, "metas": metas, "reply": reply2}
        return {**first, "first": first, "repaired": False, "metas": metas, "reply": reply}

    def solve(self, task) -> dict:
        t0 = time.time()
        skills = retrieve(self.library.active(), task_query(task), self.top_k, self.min_score) if self.library else []
        demos = self.pools[task.dataset].top(task, self.strat.k_shots) if self.strat.k_shots else []
        msgs = build_messages(task, self.strat, skills, demos)
        cands = [self._attempt(task, msgs, 0.0 if i == 0 else self.strat.temperature, i) for i in range(self.strat.n_samples)]
        # execution-result majority vote; errors never win; ties -> lowest sample index (greedy first)
        groups = defaultdict(list)
        for i, c in enumerate(cands):
            if c["err"] is None:
                groups[frozenset(c["rows"])].append(i)
        pick = min(groups.values(), key=lambda g: (-len(g), g[0]))[0] if groups else 0
        c = cands[pick]
        gold = gold_rows(task.db_path, task.gold_sql)
        gset = frozenset(gold) if gold is not None else None
        correct = c["err"] is None and gset is not None and frozenset(c["rows"]) == gset
        f = c["first"]
        first_correct = f["err"] is None and gset is not None and frozenset(f["rows"]) == gset
        metas = [m for cc in cands for m in cc["metas"]]
        rec = {
            "task_id": task.id, "dataset": task.dataset, "db_id": task.db_id, "difficulty": task.difficulty,
            "correct": bool(correct), "first_correct": bool(first_correct), "repaired": c["repaired"],
            "error": c["err"], "error_type": error_type(c["err"], correct),
            "executable": c["err"] is None, "n_rows": len(c["rows"] or []),
            "soft_f1": soft_f1(c["rows"] if c["err"] is None else [], gold or []) if gold is not None else 0.0,
            "rves": None,  # filled by a sequential timing pass after the set (see run), so workers don't skew timings
            "sql": c["sql"], "first_sql": f["sql"] if c["repaired"] else None,
            "skills": [s.key for s in skills], "demos": [d.id for d in demos],
            "n_candidates": len(cands), "vote_size": max((len(g) for g in groups.values()), default=0),
            "oracle_correct": any(cc["err"] is None and gset is not None and frozenset(cc["rows"]) == gset for cc in cands),
            "llm_calls": len(metas), "llm_seconds": round(sum(m.get("seconds") or 0 for m in metas), 2),
            "in_tokens": sum(m.get("in_tokens") or 0 for m in metas), "out_tokens": sum(m.get("out_tokens") or 0 for m in metas),
            "cached": all(m.get("cached") for m in metas), "wall_seconds": round(time.time() - t0, 2),
            # prompt + answer hit the context window: Ollama silently drops the start of the prompt
            "ctx_overflow": any((m.get("in_tokens") or 0) + self.strat.max_tokens > self.num_ctx for m in metas),
        }
        return rec

    def safe_solve(self, tid) -> dict:
        """solve(), but a task whose LLM calls keep failing is recorded as wrong (error_type llm_error)
        instead of killing the whole run; the next run retries it."""
        try:
            return self.solve(self.tasks[tid])
        except RuntimeError as e:
            t = self.tasks[tid]
            print(f"    LLM error on {tid}: {e}", flush=True)
            return {"task_id": tid, "dataset": t.dataset, "db_id": t.db_id, "difficulty": t.difficulty,
                    "correct": False, "first_correct": False, "repaired": False, "error": str(e),
                    "error_type": "llm_error", "executable": False, "n_rows": 0, "soft_f1": 0.0, "rves": 0.0,
                    "sql": "", "first_sql": None, "skills": [], "demos": [], "n_candidates": 0, "vote_size": 0,
                    "oracle_correct": False, "llm_calls": 0, "llm_seconds": 0, "in_tokens": 0, "out_tokens": 0,
                    "cached": False, "wall_seconds": 0, "ctx_overflow": False}

    def _time_rves(self, s, ids, done, pred_f):
        """Single-threaded R-VES timing for BIRD predictions that lack it; rewrites preds.jsonl once."""
        todo = [done[(s, i)] for i in ids if (s, i) in done and done[(s, i)].get("rves") is None]
        if not todo:
            return
        for r in todo:
            t = self.tasks[r["task_id"]]
            r["rves"] = rves_reward(t.db_path, r["sql"], t.gold_sql, r["correct"])
        with open(pred_f, "w") as fh:
            for r in done.values():
                fh.write(json.dumps(r, default=str) + "\n")

    # ---------------------------------------------------------------- a whole run
    def run(self, sets, out_dir: Path, limit=None, log=print):
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_f = out_dir / "preds.jsonl"
        done = {}
        if pred_f.exists():
            for line in open(pred_f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # last line cut off by a kill; that task is simply solved again
                if r.get("error_type") == "llm_error":
                    continue  # the LLM server failed on this task last time: solve it again
                done[(r["set"], r["task_id"])] = r
        meta = {"model": self.llm.model, "strategy": self.strat.to_dict(), "rung": RUNG[self.strat.name],
                "library": str(self.library_path) if self.library_path else None,
                "sets": sets, "repair": self.repair, "top_k": self.top_k, "limit": limit}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        set_ids = {s: eval_ids(self.splits, s)[:limit] if limit else eval_ids(self.splits, s) for s in sets}
        for s in sets:
            ids = set_ids[s]
            todo = [i for i in ids if (s, i) not in done]
            log(f"  [{s}] {len(ids)} tasks, {len(todo)} to solve")
            t0 = time.time()
            with ThreadPoolExecutor(self.workers) as ex, open(pred_f, "a") as fh:
                for k, rec in enumerate(ex.map(self.safe_solve, todo), 1):
                    rec["set"] = s
                    fh.write(json.dumps(rec, default=str) + "\n")
                    fh.flush()
                    done[(s, rec["task_id"])] = rec
                    if k % 25 == 0:
                        log(f"    {k}/{len(todo)}  {time.time() - t0:.0f}s")
            n_err = sum(done[(s, i)].get("error_type") == "llm_error" for i in ids if (s, i) in done)
            if n_err:
                log(f"    WARNING: {n_err} tasks failed with LLM-server errors (counted wrong); rerun to retry them")
            if self.ves and s.startswith("bird"):
                self._time_rves(s, ids, done, pred_f)
        recs = {s: [done[(s, i)] for i in set_ids[s] if (s, i) in done] for s in sets}
        metrics = {s: summarize_set(r) for s, r in recs.items()}
        (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics}, indent=2))
        return metrics


def summarize_set(recs: list[dict]) -> dict:
    if not recs:
        return {}
    n = len(recs)
    m = ex_summary([r["correct"] for r in recs])
    m.update({
        "ex_first_attempt": sum(r["first_correct"] for r in recs) / n,
        "soft_f1": sum(r["soft_f1"] for r in recs) / n,
        "executable_rate": sum(r["executable"] for r in recs) / n,
        "repair_rate": sum(r["repaired"] for r in recs) / n,
        "oracle_ex": sum(r["oracle_correct"] for r in recs) / n,
        "avg_llm_calls": sum(r["llm_calls"] for r in recs) / n,
        **_avg_known(recs, ("llm_seconds", "out_tokens", "in_tokens")),
        "errors": dict(Counter(r["error_type"] for r in recs)),
        "ctx_overflow_rate": sum(bool(r.get("ctx_overflow")) for r in recs) / n,
    })
    rv = [r["rves"] for r in recs if r.get("rves") is not None]
    if rv:
        m["rves"] = 100 * sum(math.sqrt(x) for x in rv) / len(rv)  # BIRD mini-dev: mean(sqrt(reward)) x 100
    for d, g in sorted(_group(recs, "difficulty").items()):
        m[f"ex_{d}"] = sum(r["correct"] for r in g) / len(g)
        m[f"n_{d}"] = len(g)
    return m


def _avg_known(recs, keys):
    """Mean cost over records that measured it; cache rows written before cost logging report 0."""
    known = [r for r in recs if r.get("in_tokens")]
    return {f"avg_{k}": (sum(r[k] for r in known) / len(known) if known else None) for k in keys}


def _group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag, or mlx:<path>[@host:port] for mlx_lm.server")
    ap.add_argument("--strategy", default="zs_repo", choices=list(LADDER))
    ap.add_argument("--ladder", action="store_true", help="run every rung P0..P8 in order")
    ap.add_argument("--rungs", help="comma list of strategies (subset of the ladder)")
    ap.add_argument("--sets", default="bird_val,bird_test,spider_test")
    ap.add_argument("--library", help="skill library JSON from a SkillEvo run (combined method)")
    ap.add_argument("--limit", type=int, help="first N tasks per set (smoke test)")
    ap.add_argument("--tag")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--no-ves", action="store_true", help="skip R-VES timing")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="Ollama context. Not part of the cache key: keep it at 16384 for every bench run")
    a = ap.parse_args()
    names = list(LADDER) if a.ladder else (a.rungs.split(",") if a.rungs else [a.strategy])
    sets = a.sets.split(",")
    for name in names:
        rn = run_name(a.model, name, a.library, a.tag or (f"n{a.limit}" if a.limit else None))
        print(f"== {rn}", flush=True)
        b = Bench(a.model, name, a.library, num_ctx=a.num_ctx, repair=not a.no_repair, ves=not a.no_ves, workers=a.workers)
        m = b.run(sets, ROOT / "runs/bench" / rn, a.limit, log=lambda x: print(x, flush=True))
        for s, v in m.items():
            if v:
                print(f"   {s:<12} EX={v['ex']:.3f} [{v['ex_ci_lo']:.3f},{v['ex_ci_hi']:.3f}] soft-F1={v['soft_f1']:.3f} "
                      f"exec={v['executable_rate']:.2f} " + (f"R-VES={v['rves']:.1f}" if "rves" in v else ""), flush=True)


if __name__ == "__main__":
    main()
