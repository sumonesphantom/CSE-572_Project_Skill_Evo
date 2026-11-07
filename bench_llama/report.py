"""Build the results table (EX + extra metrics + McNemar vs the team baseline) as markdown."""
import json, os
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SETS = ["val", "spider_val", "bird_test", "spider_transfer"]

def load_rows(p):
    return {json.loads(l)["task_id"]: bool(json.loads(l)["correct"]) for l in open(p)}

def mcnemar(a, b):
    """Exact two-sided McNemar p-value on the questions both runs answered."""
    ids = a.keys() & b.keys()
    n01 = sum(1 for i in ids if not a[i] and b[i]); n10 = sum(1 for i in ids if a[i] and not b[i])
    n = n01 + n10
    if n == 0: return 1.0, n01, n10
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n), n01, n10

base = {s: {} for s in SETS}
for l in open(ROOT / "results/llama1b/baseline_noskill/trajectories.jsonl"):
    r = json.loads(l); base[r["phase"]][r["task_id"]] = bool(r["correct"])
base_acc = json.loads(open(ROOT / "results/llama1b/baseline_noskill/rounds.jsonl").readline())["acc"]

RUNS = [("Team baseline (repo, no skills)", None),
        ("repo pipeline, reimplemented", "llama3.2_1b__repo"),
        ("+ robust SQL extraction", "llama3.2_1b__extract"),
        ("+ compact full schema", "llama3.2_1b__compact"),
        ("+ schema linking (pruned)", "llama3.2_1b__pruned"),
        ("+ 3 few-shot examples", "llama3.2_1b__fewshot"),
        ("+ 1 few-shot example", "llama3.2_1b__fewshot1"),
        ("+ few-shot + 5-vote", "llama3.2_1b__sc5"),
        ("**pruned + 5-vote (sc5_nofs) — chosen**", "llama3.2_1b__sc5_nofs"),
        ("Fine-tuned (LoRA), pruned prompt", "llama3.2-1b-sql__pruned"),
        ("Fine-tuned (LoRA), pruned + 5-vote", "llama3.2-1b-sql__sc5_nofs")]

out = ["| Setup | BIRD val | Spider val | **BIRD test** | **Spider test** | sec/q | LLM calls/q |", "|---|---|---|---|---|---|---|"]
detail = []
for name, d in RUNS:
    cells, secs, calls = [], [], []
    for s in SETS:
        if d is None:
            cells.append(f"{100*base_acc[s]:.1f}" if s in base_acc else "–"); continue
        p = ROOT / f"results/llama1b/runs_bench/{d}/{s}.summary.json"
        if not p.exists(): cells.append("–"); continue
        j = json.loads(p.read_text()); secs.append(j["sec_per_q"]); calls.append(j["calls_per_q"])
        cell = f"{100*j['EX']:.1f}"
        if s in ("bird_test", "spider_transfer") and base[s]:
            pv, w, l_ = mcnemar(base[s], load_rows(ROOT / f"results/llama1b/runs_bench/{d}/{s}.jsonl"))
            cell += f" (p={pv:.3f})"
            detail.append((name, s, j))
        cells.append(cell)
    out.append(f"| {name} | " + " | ".join(cells) + f" | {sum(secs)/len(secs):.1f} | {sum(calls)/len(calls):.1f} |" if secs
               else f"| {name} | " + " | ".join(cells) + " | – | – |")

ext = ["| Setup | Set | EX % | Valid SQL % | Simple | Moderate | Challenging | Errors (top) |", "|---|---|---|---|---|---|---|---|"]
for name, s, j in detail:
    bd = j.get("by_difficulty") or {}
    oc = sorted(((k, v) for k, v in j["outcomes"].items() if k != "correct"), key=lambda x: -x[1])[:3]
    f = lambda k: f"{100*bd[k]:.1f}" if k in bd else "–"  # noqa: E731
    ext.append(f"| {name} | {s} | {100*j['EX']:.1f} | {100*j['valid_sql']:.0f} | {f('simple')} | {f('moderate')} | {f('challenging')} | "
               + ", ".join(f"{k} {v}" for k, v in oc) + " |")
print("\n".join(out)); print(); print("\n".join(ext))
