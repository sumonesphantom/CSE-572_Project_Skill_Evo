"""Evaluate solver variants on fixed sets.

    python -m bench_llama.evaluate --variants repo,extract,compact --sets val,spider_val
    python -m bench_llama.evaluate --variants best --sets bird_test,spider_transfer

Tune on val / spider_val only; bird_test and spider_transfer are the shared reporting sets
(identical to the repo's: full bird:test and the same seed-0 sample of 200 spider:test).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillevo.data import load_tasks  # noqa: E402

from bench_llama.client import Client  # noqa: E402
from bench_llama.solver import FewShot, Variant, solve  # noqa: E402

MODEL = "llama3.2:1b"

BASE = Variant("repo")                                     # the team repo's pipeline, reimplemented 1:1
VARIANTS = {}
for v in [
    BASE,
    replace(BASE, name="extract"),                                                   # + robust SQL extraction
    replace(BASE, name="compact", fast=True, schema="compact", prompt="improved"),              # + full compact schema, new prompt
    replace(BASE, name="pruned", fast=True, schema="pruned", prompt="improved"),                # + schema linking (top-4 tables)
    replace(BASE, name="fewshot", fast=True, schema="pruned", prompt="improved", fewshot=3),    # + 3 retrieved train examples
    replace(BASE, name="sc5", fast=True, schema="pruned", prompt="improved", fewshot=3, samples=5),   # + 5-sample vote
    # follow-ups after the first tuning pass (few-shot hurt Spider):
    replace(BASE, name="fewshot1", fast=True, schema="pruned", prompt="improved", fewshot=1),    # 1 example instead of 3
    replace(BASE, name="sc5_nofs", fast=True, schema="pruned", prompt="improved", samples=5),    # vote, no examples
]:
    VARIANTS[v.name] = v
VARIANTS["repo"] = replace(BASE, robust_extract=False)


def load_sets(seed: int = 0):
    tasks = {**load_tasks(ROOT / "data/processed/bird.jsonl"), **load_tasks(ROOT / "data/processed/spider.jsonl")}
    sp = json.loads((ROOT / "data/processed/splits.json").read_text())
    sample = lambda ids, n: sorted(random.Random(seed).sample(list(ids), n)) if len(ids) > n else list(ids)  # noqa: E731
    sets = {"val": sp["bird"]["val"], "spider_val": sample(sp["spider"]["val"], 100),
            "bird_test": sp["bird"]["test"], "spider_transfer": sample(sp["spider"]["test"], 200)}
    # BIRD mini-dev has 2 duplicated questions (ids 137, 138) that the shared splits put in
    # two splits at once; drop anything that is also an evaluation item from the example pool.
    held_out = set().union(*sets.values()) | set(sp["bird"]["test"]) | set(sp["spider"]["test"])
    train = [tasks[i] for i in sp["bird"]["train"] + sp["spider"]["train"] if i not in held_out]
    return tasks, sets, train


def err_cat(e):
    if not e:
        return None
    e = e.lower()
    for key, name in (("no such column", "wrong column"), ("no such table", "wrong table"), ("syntax", "syntax"),
                      ("unrecognized token", "syntax"), ("incomplete input", "syntax"), ("timed out", "timeout")):
        if key in e:
            return name
    return "other error"


def summarize(rows, tasks):
    n = len(rows)
    by = defaultdict(list)
    for r in rows:
        by[tasks[r["task_id"]].difficulty].append(r["correct"])
    out = Counter(err_cat(r["error"]) or ("correct" if r["correct"] else "wrong result") for r in rows)
    return {"n": n, "EX": sum(r["correct"] for r in rows) / n,
            "valid_sql": sum(r["error"] is None for r in rows) / n,
            "by_difficulty": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())},
            "outcomes": dict(out.most_common()),
            "sec_per_q": sum(r["gen_secs"] for r in rows) / n, "calls_per_q": sum(r["calls"] for r in rows) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--sets", default="val,spider_val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/llama1b/runs_bench")
    a = ap.parse_args()
    tasks, sets, train = load_sets()
    fs = FewShot(train)
    client = Client(a.model, str(ROOT / "data/cache/bench_cache.sqlite"), max_tokens=512)
    for vname in a.variants.split(","):
        v = VARIANTS[vname]
        for sname in a.sets.split(","):
            ids = sets[sname]
            t0 = time.time()
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(lambda i: solve(tasks[i], v, client, fs), ids))
            rows = [asdict(r) for r in res]
            d = ROOT / a.out / f"{a.model.replace(':', '_')}__{vname}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{sname}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = summarize(rows, tasks)
            s.update(variant=vname, set=sname, model=a.model, config=asdict(v), wall_secs=round(time.time() - t0, 1))
            (d / f"{sname}.summary.json").write_text(json.dumps(s, indent=2))
            print(f"{vname:10} {sname:16} EX={s['EX']:.1%} valid={s['valid_sql']:.0%} "
                  f"{s['sec_per_q']:.1f}s/q  wall={s['wall_secs']:.0f}s  {s['outcomes']}", flush=True)


if __name__ == "__main__":
    main()
