"""Benchmark one local model with one or more prompt strategies on fixed BIRD / Spider sets.

    python -m phibench.run --model phi4-mini --strategies a1_repo                     # bird_val only
    python -m phibench.run --model phi4-mini --strategies a1_repo,a2_mschema --sets bird_val,bird_test,spider_test --allow-test
    python -m phibench.run --model phi4-mini --strategies a1_repo --limit 5 --tag smoke  # quick check
    python -m phibench.run --backend openai --base-url http://localhost:8080 \\
        --model mlx-community/Phi-4-mini-instruct-4bit --label phi4mini-r16 --strategies a3_rules

Test sets are sealed by default: bird_test / spider_test need --allow-test, so decisions are
made on bird_val. Outputs: runs/phibench/<run>/{meta.json, preds.jsonl, metrics.json}. A rerun
resumes from preds.jsonl and the response cache (data/cache/phibench_cache.sqlite).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skillevo.data import load_tasks  # noqa: E402
from skillevo.sqlexec import execute  # noqa: E402

from phibench.embed import Embedder  # noqa: E402
from phibench.examples import build_pools  # noqa: E402
from phibench.llm import Cache, ChatClient  # noqa: E402
from phibench.metrics import error_type, ex_summary, gold_rows, rves_reward, soft_f1  # noqa: E402
from phibench.parsing import extract_sql  # noqa: E402
from phibench.repair import EMPTY_MESSAGE, error_message  # noqa: E402
from phibench.values import empty_value_hints  # noqa: E402
from phibench.strategies import STRATEGIES, build_messages  # noqa: E402

EVAL_SETS = {  # name -> (dataset, split, max); spider_test = the same seeded 200 as config.yaml
    "bird_val": ("bird", "val", None),
    "bird_test": ("bird", "test", None),
    "spider_test": ("spider", "test", 200),
}
TEST_SETS = {"bird_test", "spider_test"}
SPIDER_TRAIN = ROOT / "data/raw/spider/spider_data/train_spider.json"
REPAIR_MSG = "That query failed with: {err}\nFix it. Return only the corrected ```sql block."


def eval_ids(splits: dict, name: str, seed: int = 0) -> list[str]:
    ds, sp, mx = EVAL_SETS[name]
    ids = list(splits[ds][sp])
    if mx and len(ids) > mx:  # same rule as skillevo.runner.Experiment._split
        ids = sorted(random.Random(seed).sample(ids, mx))
    return ids


def run_name(label: str, strategy: str, tag: str | None = None) -> str:
    parts = [re.sub(r"[^A-Za-z0-9.]+", "-", label).strip("-"), strategy] + ([tag] if tag else [])
    return "__".join(parts)


def load_data(root: Path = ROOT):
    tasks = {}
    for f in ("bird", "spider"):
        p = root / f"data/processed/{f}.jsonl"
        if p.exists():
            tasks.update(load_tasks(p))
    splits = json.loads((root / "data/processed/splits.json").read_text())
    return tasks, splits


class Bench:
    def __init__(self, client, strategy: str, tasks: dict, splits: dict, pools=None,
                 workers: int = 2, ves: bool = True, sql_timeout: float = 30.0):
        self.client, self.strat = client, STRATEGIES[strategy]
        self.tasks, self.splits, self.pools = tasks, splits, pools or {}
        self.workers, self.ves, self.sql_timeout = workers, ves, sql_timeout
        self.wall: dict[str, dict] = {}
        if self.strat.k_shots and not self.pools:
            raise ValueError(f"{strategy} needs few-shot pools")

    # ------------------------------------------------------------- one question
    def _run_sql(self, task, sql):
        rows, err = execute(task.db_path, sql, self.sql_timeout)
        gold = gold_rows(task.db_path, task.gold_sql)
        ok = err is None and gold is not None and frozenset(rows) == frozenset(gold)
        return rows, err, ok, gold

    def solve(self, task) -> dict:
        t0 = time.time()
        demos = self.pools[task.dataset].top(task, self.strat.k_shots, self.strat.retriever) if self.strat.k_shots else []
        msgs = build_messages(task, self.strat, demos)
        reply, meta = self.client.chat(msgs)
        metas = [meta]
        sql = extract_sql(reply)
        rows, err, ok, gold = self._run_sql(task, sql)
        first = {"sql": sql, "err": err, "ok": ok}
        kinds: list[str] = []
        hist = msgs
        while self.strat.repair != "none" and len(kinds) < self.strat.repair_rounds:
            if err:
                kind = "error"
                fb = (error_message(task.db_path, sql, err) if self.strat.repair == "schema"
                      else REPAIR_MSG.format(err=err))
            elif self.strat.retry_empty and not rows:
                kind, fb = "empty", EMPTY_MESSAGE
                if self.strat.value_hints:
                    extra = empty_value_hints(task.db_path, sql)
                    fb = EMPTY_MESSAGE + ("\n" + extra if extra else "")
            else:
                break
            hist = hist + [{"role": "assistant", "content": reply}, {"role": "user", "content": fb}]
            reply, m2 = self.client.chat(hist)
            metas.append(m2)
            sql = extract_sql(reply)
            rows, err, ok, gold = self._run_sql(task, sql)
            kinds.append(kind)
        repaired = bool(kinds)
        n_rows = len(rows) if rows is not None else 0
        return {
            "task_id": task.id, "dataset": task.dataset, "db_id": task.db_id, "difficulty": task.difficulty,
            "correct": bool(ok), "first_correct": bool(first["ok"]), "repaired": repaired,
            "executable": err is None, "error": err, "error_type": error_type(err, ok, n_rows), "n_rows": n_rows,
            "soft_f1": soft_f1(rows if err is None else [], gold or []) if gold is not None else 0.0,
            "rves": None, "sql": sql, "first_sql": first["sql"] if repaired else None, "reply": reply,
            "repair_kinds": kinds, "demos": [d.id for d in demos], "llm_calls": len(metas),
            "llm_seconds": round(sum(m.get("seconds") or 0 for m in metas), 3),
            "in_tokens": sum(m.get("in_tokens") or 0 for m in metas),
            "out_tokens": sum(m.get("out_tokens") or 0 for m in metas),
            "cached": all(m.get("cached") for m in metas), "wall_seconds": round(time.time() - t0, 3),
        }

    # ------------------------------------------------------------- a whole run
    def run(self, sets: list[str], out_dir: Path, limit: int | None = None,
            allow_test: bool = False, log=print) -> dict:
        sealed = [s for s in sets if s in TEST_SETS and not allow_test]
        if sealed:
            raise PermissionError(f"test sets {sealed} are sealed; pass --allow-test only for final or pre-registered runs")
        unknown = [s for s in sets if s not in EVAL_SETS]
        if unknown:
            raise ValueError(f"unknown sets {unknown}; choose from {list(EVAL_SETS)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"model": self.client.model, "label": self.client.label, "backend": self.client.backend,
                "num_ctx": self.client.num_ctx, "strategy": self.strat.to_dict(), "limit": limit}
        mf = out_dir / "meta.json"
        if mf.exists():
            old = json.loads(mf.read_text())
            for k in ("model", "label", "backend", "num_ctx", "strategy"):
                if old.get(k) != meta[k]:
                    raise ValueError(f"{out_dir} was made with a different {k}: {old.get(k)} != {meta[k]}")
            meta["sets"] = sorted(set(old.get("sets", [])) | set(sets))
        else:
            meta["sets"] = sets
        meta["allow_test_used"] = bool(set(meta["sets"]) & TEST_SETS)
        mf.write_text(json.dumps(meta, indent=2))
        pred_f = out_dir / "preds.jsonl"
        done = {}
        if pred_f.exists():
            for line in open(pred_f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a kill: solved again
                done[(r["set"], r["task_id"])] = r
        set_ids = {s: eval_ids(self.splits, s)[:limit] if limit else eval_ids(self.splits, s) for s in sets}
        for s in sets:
            ids = set_ids[s]
            todo = list(dict.fromkeys(i for i in ids if (s, i) not in done))
            log(f"  [{s}] {len(ids)} questions, {len(todo)} to solve")
            t0 = time.time()
            with ThreadPoolExecutor(self.workers) as ex, open(pred_f, "a") as fh:
                for k, rec in enumerate(ex.map(lambda tid: self.solve(self.tasks[tid]), todo), 1):
                    rec["set"] = s
                    fh.write(json.dumps(rec, default=str) + "\n")
                    fh.flush()
                    done[(s, rec["task_id"])] = rec
                    if k % 20 == 0 or k == len(todo):
                        log(f"    {k}/{len(todo)}  {time.time() - t0:.0f}s")
            if todo:
                self.wall[s] = {"solved": len(todo), "seconds": round(time.time() - t0, 1)}
            if self.ves and s.startswith("bird"):
                self._time_rves(s, ids, done, pred_f)
        recs = {s: [done[(s, i)] for i in set_ids[s] if (s, i) in done] for s in sets}
        metrics = {s: summarize(r) for s, r in recs.items()}
        for s, w in self.wall.items():  # real throughput, only for sets solved in this invocation
            if metrics.get(s):
                metrics[s]["wall_seconds_per_question"] = round(w["seconds"] / w["solved"], 2)
        old_metrics = {}
        mfile = out_dir / "metrics.json"
        if mfile.exists():
            old_metrics = json.loads(mfile.read_text()).get("metrics", {})
        mfile.write_text(json.dumps({"meta": meta, "metrics": {**old_metrics, **metrics}}, indent=2))
        return metrics

    def _time_rves(self, s, ids, done, pred_f):
        todo = [done[(s, i)] for i in dict.fromkeys(ids) if (s, i) in done and done[(s, i)].get("rves") is None]
        if not todo:
            return
        for r in todo:  # sequential, so concurrent queries don't distort the timings
            t = self.tasks[r["task_id"]]
            r["rves"] = rves_reward(t.db_path, r["sql"], t.gold_sql, r["correct"])
        with open(pred_f, "w") as fh:
            for r in done.values():
                fh.write(json.dumps(r, default=str) + "\n")


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {}
    n = len(recs)
    m = ex_summary([r["correct"] for r in recs])
    known = [r for r in recs if r.get("in_tokens")]
    m.update({
        "ex_first": sum(r["first_correct"] for r in recs) / n,
        "soft_f1": sum(r["soft_f1"] for r in recs) / n,
        "executable_rate": sum(r["executable"] for r in recs) / n,
        "repair_rate": sum(r["repaired"] for r in recs) / n,
        "avg_llm_seconds": (sum(r["llm_seconds"] for r in known) / len(known)) if known else None,
        "avg_in_tokens": (sum(r["in_tokens"] for r in known) / len(known)) if known else None,
        "avg_out_tokens": (sum(r["out_tokens"] for r in known) / len(known)) if known else None,
        "errors": dict(Counter(r["error_type"] for r in recs)),
    })
    rv = [r["rves"] for r in recs if r.get("rves") is not None]
    if rv:
        m["rves"] = 100 * sum(x ** 0.5 for x in rv) / len(rv)
    by = defaultdict(list)
    for r in recs:
        by[r["difficulty"]].append(r["correct"])
    for d, v in sorted(by.items()):
        m[f"ex_{d}"], m[f"n_{d}"] = sum(v) / len(v), len(v)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi4-mini")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--base-url")
    ap.add_argument("--label", help="cache/run namespace (required to tell LoRA adapters apart)")
    ap.add_argument("--strategies", default="a1_repo", help=f"comma list from {list(STRATEGIES)}")
    ap.add_argument("--sets", default="bird_val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--think", choices=["on", "off"], help="send Ollama's think flag (default: not sent)")
    ap.add_argument("--no-ves", action="store_true")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="local Ollama embedding model (b3_embed)")
    a = ap.parse_args(argv)
    tasks, splits = load_data()
    sets = a.sets.split(",")
    names = a.strategies.split(",")
    bad = [n for n in names if n not in STRATEGIES]
    if bad:
        ap.error(f"unknown strategies {bad}")
    cache = Cache(ROOT / "data/cache/phibench_cache.sqlite")
    pools = None
    for name in names:
        strat = STRATEGIES[name]
        if strat.k_shots and pools is None:
            ev = {i for s in EVAL_SETS for i in eval_ids(splits, s)}
            emb = Embedder(a.embed_model, a.base_url if a.backend == "ollama" and a.base_url else "http://localhost:11434",
                           ROOT / "data/cache/phibench_embeddings.sqlite")
            pools = build_pools(tasks, splits, SPIDER_TRAIN, ev, embedder=emb)
        client = ChatClient(a.model, a.backend, a.base_url, a.label, cache, a.num_ctx, strat.max_tokens,
                            think=None if a.think is None else a.think == "on")
        rn = run_name(client.label, name, a.tag or (f"n{a.limit}" if a.limit else None))
        print(f"== {rn}", flush=True)
        b = Bench(client, name, tasks, splits, pools, a.workers, ves=not a.no_ves)
        m = b.run(sets, ROOT / "runs/phibench" / rn, a.limit, a.allow_test, log=lambda x: print(x, flush=True))
        for s, v in m.items():
            if v:
                print(f"   {s:<12} EX={v['ex']:.3f} [{v['ex_ci_lo']:.3f}, {v['ex_ci_hi']:.3f}]  first={v['ex_first']:.3f}  "
                      f"soft-F1={v['soft_f1']:.3f}  exec={v['executable_rate']:.2f}  errors={v['errors']}"
                      + (f"  R-VES={v['rves']:.1f}" if "rves" in v else ""), flush=True)


if __name__ == "__main__":
    main()
