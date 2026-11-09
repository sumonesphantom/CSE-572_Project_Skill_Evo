"""Gemma benchmark: validation ladder -> selection -> test sets -> report. Resumable: rerun to continue.

    python scripts/gemma_pipeline.py                         # gemma4:e2b-it-q4_K_M, all phases
    python scripts/gemma_pipeline.py --phase val             # validation ladder only

Every setup is chosen on the validation sets (bird_val 98, spider_val 100). The test sets
(bird_test 155, spider_test 200) are run only for the baseline and the setups picked on validation.
Writes results/gemma4_e2b/{bench_results.csv, tables.md, selection.json, runs_bench/}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.run import run_name  # noqa: E402

RUNGS = ["zs_repo", "sm_extract", "sm_compact", "sm_pruned", "sm_pruned_fs3",
         "zs_mschema", "ms_rules", "ms_fs3", "sm_pruned_sc5"]   # the 5-vote rung costs 5x, so it runs last
VAL = ["bird_val", "spider_val"]
TEST = ["bird_val", "spider_val", "bird_test", "spider_test"]
HOST = "http://localhost:11434"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_ollama():
    try:
        requests.get(f"{HOST}/api/version", timeout=5)
        return
    except requests.RequestException:
        pass
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": "2", "OLLAMA_KEEP_ALIVE": "-1"}
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open(ROOT / "runs/ollama.log", "a"),
                     stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            requests.get(f"{HOST}/api/version", timeout=5)
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("ollama did not start")


def metrics(model, rung):
    f = ROOT / "runs/bench" / run_name(model, rung) / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8"))["metrics"] if f.exists() else {}


def complete(model, rung, sets):
    m = metrics(model, rung)
    return all(m.get(s) and not m[s].get("errors", {}).get("llm_error") for s in sets)


def bench(model, rung, sets, num_ctx, workers):
    for attempt in range(3):   # a rerun retries questions that hit an Ollama server error
        if complete(model, rung, sets):
            return
        ensure_ollama()
        log(f"RUN {rung} on {','.join(sets)} (attempt {attempt + 1})")
        subprocess.run([sys.executable, "-m", "bench.run", "--model", model, "--strategy", rung,
                        "--sets", ",".join(sets), "--num-ctx", str(num_ctx), "--workers", str(workers)], cwd=ROOT)
    log(f"WARNING {rung} still incomplete on {sets}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e2b-it-q4_K_M")
    ap.add_argument("--num-ctx", type=int, default=16384, help="same as the Qwen runs; not part of the cache key")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--phase", default="all", choices=["val", "all"])
    ap.add_argument("--out", default="results/gemma4_e2b")
    a = ap.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)

    log("PHASE validation", a.model)
    for r in RUNGS:
        bench(a.model, r, VAL, a.num_ctx, a.workers)
        m = metrics(a.model, r)
        if m.get("bird_val") and m.get("spider_val"):
            log(f"VAL {r:<16} bird_val {100 * m['bird_val']['ex']:.1f}  spider_val {100 * m['spider_val']['ex']:.1f}  "
                f"s/q {m['bird_val'].get('avg_llm_seconds') or 0:.1f}")

    val = {r: metrics(a.model, r) for r in RUNGS}
    val = {r: m for r, m in val.items() if m.get("bird_val") and m.get("spider_val")}
    bird = lambda r: val[r]["bird_val"]["ex"]      # noqa: E731
    spider = lambda r: val[r]["spider_val"]["ex"]  # noqa: E731
    best_bird, best_spider = max(val, key=bird), max(val, key=spider)   # ties -> the earlier (simpler) rung
    best_mean = max(val, key=lambda r: (bird(r) + spider(r)) / 2)
    final = list(dict.fromkeys(["zs_repo", best_bird, best_spider, best_mean]))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(
        {"model": a.model, "picked": {"bird": best_bird, "spider": best_spider, "overall": best_mean},
         "final_runs": final, "validation": {r: {"bird_val": bird(r), "spider_val": spider(r)} for r in val}},
        indent=2), encoding="utf-8")
    log("PICKED", "bird:", best_bird, "| spider:", best_spider, "| overall:", best_mean)
    if a.phase == "val":
        return

    log("PHASE test", final)
    for r in final:
        bench(a.model, r, TEST, a.num_ctx, a.workers)

    log("PHASE report")
    subprocess.run([sys.executable, "-m", "bench.smol_report", "--prefix", a.model.split(":")[0], "--out", str(out)],
                   cwd=ROOT)
    log("DONE")


if __name__ == "__main__":
    main()
