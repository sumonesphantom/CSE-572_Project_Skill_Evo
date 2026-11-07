"""Export the Llama 3.2 1B results to results/llama1b/bench_results.csv (one row per setup and set).

Re-executes the saved SQL (needs data/raw + data/processed from scripts/prepare_data.py); no LLM calls.
Columns this pipeline does not measure (token counts, R-VES) are left empty.
    python -m bench_llama.export_csv [--data-root PATH]
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skillevo.data import load_tasks  # noqa: E402
from skillevo.monitor import bootstrap_ci  # noqa: E402
from skillevo.sqlexec import execute  # noqa: E402

COLS = ("run,model,strategy,rung,library,set,n,ex,ex_ci_lo,ex_ci_hi,ex_first_attempt,soft_f1,executable_rate,"
        "repair_rate,oracle_ex,avg_llm_calls,avg_llm_seconds,avg_out_tokens,avg_in_tokens,rves,ex_challenging,"
        "n_challenging,ex_moderate,n_moderate,ex_simple,n_simple").split(",")
SET_NAMES = {"val": "bird_val", "spider_val": "spider_val", "bird_test": "bird_test", "spider_transfer": "spider_test"}
# Step labels (see docs/LLAMA_BENCHMARK.md). L0 "repo" is the repo prompt + repair turn.
RUNG = {"repo": "L0", "extract": "L1", "compact": "L2", "pruned": "L3", "fewshot": "L4", "fewshot1": "L4b",
        "sc5": "L5", "sc5_nofs": "L5b"}


def soft_f1(pred, gold) -> float:
    """BIRD mini-dev `calculate_f1_score` (port of evaluation_f1.py)."""
    if not pred and not gold:
        return 1.0
    pred, gt = list(dict.fromkeys(pred or [])), list(dict.fromkeys(gold))
    m = p = t = 0.0
    for i, g in enumerate(gt):
        if i >= len(pred):
            t += 1; continue  # noqa: E702
        tot = len(g) or 1
        hit = sum(1 for v in pred[i] if v in g)
        m += hit / tot; p += (len(pred[i]) - hit) / tot; t += sum(1 for v in g if v not in pred[i]) / tot  # noqa: E702
    p += max(0, len(pred) - len(gt))
    prec = m / (m + p) if m + p else 0.0
    rec = m / (m + t) if m + t else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


_gold = {}
def gold_rows(task):
    if task.id not in _gold:
        rows, err = execute(task.db_path, task.gold_sql, 120)
        _gold[task.id] = None if err else list(rows)
    return _gold[task.id]


def correct(task, sql):
    rows, err = execute(task.db_path, sql, 30)
    g = gold_rows(task)
    return (err is None and g is not None and frozenset(rows) == frozenset(g)), (None if err else list(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT), help="folder holding data/processed and data/raw")
    a = ap.parse_args()
    dr = Path(a.data_root)
    tasks = {}
    for f in ("bird.jsonl", "spider.jsonl"):
        for t in load_tasks(dr / "data/processed" / f).values():
            t.db_path = str(dr / t.db_path) if not Path(t.db_path).is_absolute() else t.db_path
            tasks[t.id] = t
    out = []
    for d in sorted((ROOT / "results/llama1b/runs_bench").iterdir()):
        model, variant = d.name.split("__")
        for s, sname in SET_NAMES.items():
            p = d / f"{s}.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.open()]
            ex, first, oracle, f1, diff = [], [], [], [], {}
            for r in rows:
                t = tasks[r["task_id"]]
                ok, pred = correct(t, r["sql"])
                ex.append(ok)
                cands = r.get("candidates") or [r["sql"]]
                cok = [correct(t, c)[0] for c in cands]
                first.append(cok[0]); oracle.append(any(cok) or ok)
                g = gold_rows(t)
                f1.append(soft_f1(pred, g) if (pred is not None and g is not None) else 0.0)
                diff.setdefault(t.difficulty, []).append(ok)
            exa = np.array(ex, float)
            lo, hi = bootstrap_ci(exa)
            row = dict(run=d.name, model=model.replace("_", ":", 1) if model == "llama3.2_1b" else model,
                       strategy=variant, rung=RUNG.get(variant, "") + ("+FT" if model.endswith("-sql") else ""),
                       library="", set=sname, n=len(rows), ex=exa.mean(), ex_ci_lo=lo, ex_ci_hi=hi,
                       ex_first_attempt=np.mean(first), soft_f1=np.mean(f1),
                       executable_rate=np.mean([r["error"] is None for r in rows]),
                       repair_rate=np.mean([bool(r.get("repaired")) for r in rows]), oracle_ex=np.mean(oracle),
                       avg_llm_calls=np.mean([r["calls"] for r in rows]),
                       avg_llm_seconds=np.mean([r["gen_secs"] for r in rows]),
                       avg_out_tokens="", avg_in_tokens="", rves="")
            for k in ("challenging", "moderate", "simple"):
                v = diff.get(k, [])
                row[f"ex_{k}"], row[f"n_{k}"] = (np.mean(v) if v else ""), len(v)
            out.append(row)
            print(f"{d.name:28s} {sname:12s} EX={100*row['ex']:.1f}  first={100*row['ex_first_attempt']:.1f}  "
                  f"oracle={100*row['oracle_ex']:.1f}  softF1={100*row['soft_f1']:.1f}")
    dst = ROOT / "results/llama1b/bench_results.csv"
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(out)  # noqa: E702
    print("wrote", dst.relative_to(ROOT), len(out), "rows")


if __name__ == "__main__":
    main()
