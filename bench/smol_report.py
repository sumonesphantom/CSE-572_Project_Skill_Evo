"""SmolLM2 results: team CSV, paired tests and markdown tables from runs/bench/.

    python -m bench.smol_report                      # every run whose model starts with "smollm2"
    python -m bench.smol_report --prefix smollm2,mlx-smollm2    # also fine-tuned runs

Writes
  results/smollm2/bench_results.csv   same columns as the team's results/bench_results.csv (+ ctx_overflow_rate)
  results/smollm2/tables.md           results table, paired tests vs the baseline and vs the parent rung
  results/smollm2/runs_bench/<run>/   metrics.json + preds.jsonl (per-question outputs)
Runs made with --limit (smoke tests) are skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.strategies import RUNG  # noqa: E402
from skillevo.monitor import mcnemar  # noqa: E402

OUT = ROOT / "results/smollm2"
TEAM_COLS = ["run", "model", "strategy", "rung", "library", "set", "n", "ex", "ex_ci_lo", "ex_ci_hi",
             "ex_first_attempt", "soft_f1", "executable_rate", "repair_rate", "oracle_ex", "avg_llm_calls",
             "avg_llm_seconds", "avg_out_tokens", "avg_in_tokens", "rves", "ex_challenging", "n_challenging",
             "ex_moderate", "n_moderate", "ex_simple", "n_simple"]
EXTRA_COLS = ["ctx_overflow_rate"]
SETS = ["bird_val", "spider_val", "bird_test", "spider_test"]
BASELINE = "zs_repo"
# the rung each SmolLM2 strategy changes one thing from
PARENT = {"sm_extract": "zs_repo", "sm_compact": "sm_extract", "sm_pruned": "sm_compact",
          "sm_pruned_rules": "sm_pruned", "sm_pruned_fs1": "sm_pruned", "sm_pruned_fs3": "sm_pruned",
          "sm_pruned_sc5": "sm_pruned", "sm_pruned_fs1_sc5": "sm_pruned_fs1", "sm_pruned_fs3_sc5": "sm_pruned_fs3",
          "zs_mschema": "zs_repo", "ms_fs3": "zs_mschema"}


def load(prefixes, smoke=False):
    rows, preds = [], {}
    for d in sorted((ROOT / "runs/bench").glob("*")):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        j = json.loads(mf.read_text())
        meta = j["meta"]
        if (meta.get("limit") and not smoke) or not any(meta["model"].startswith(p) for p in prefixes):
            continue
        p = defaultdict(dict)
        for line in open(d / "preds.jsonl"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p[r["set"]][r["task_id"]] = r
        preds[d.name] = p
        st = meta["strategy"]["name"]
        for s, m in j["metrics"].items():
            if m:
                rows.append({"run": d.name, "model": meta["model"], "strategy": st,
                             "rung": meta.get("rung") or RUNG.get(st, ""), "library": "", "set": s, **m})
    return pd.DataFrame(rows), preds


def paired(preds, a, b, s):
    pa, pb = preds.get(a, {}).get(s, {}), preds.get(b, {}).get(s, {})
    ids = sorted(set(pa) & set(pb))
    return mcnemar([pa[i]["correct"] for i in ids], [pb[i]["correct"] for i in ids]) if ids else None


def pct(x):
    return "–" if x is None or pd.isna(x) else f"{100 * x:.1f}"


def md(rows, cols):
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows])


def tables(df, preds):
    out = []
    # 1. EX per run and set
    rows = []
    for (run, model, st, rung), g in df.groupby(["run", "model", "strategy", "rung"], sort=False):
        by = g.set_index("set")
        r = {"Run": run, "Model": model, "Rung": rung, "Strategy": st}
        for s in SETS:
            r[s] = pct(by["ex"].get(s)) if s in by.index else "–"
        bt = by.loc["bird_test"] if "bird_test" in by.index else None
        r["BIRD test Soft-F1"] = pct(bt["soft_f1"]) if bt is not None else "–"
        r["BIRD test R-VES"] = f"{bt['rves']:.1f}" if bt is not None and pd.notna(bt.get("rves")) else "–"
        r["ctx overflow"] = pct(g["ctx_overflow_rate"].max()) if "ctx_overflow_rate" in g else "–"
        r["s/q (BIRD val)"] = (f"{by.loc['bird_val', 'avg_llm_seconds']:.1f}"
                               if "bird_val" in by.index and pd.notna(by.loc["bird_val"].get("avg_llm_seconds")) else "–")
        rows.append(r)
    rows.sort(key=lambda r: (r["Model"], r["Rung"]))
    out.append("## EX % per run\n\n" + md(rows, ["Model", "Rung", "Strategy", *SETS, "BIRD test Soft-F1",
                                                "BIRD test R-VES", "ctx overflow", "s/q (BIRD val)"]))
    # 2. paired tests: vs baseline and vs parent rung (same model)
    rows = []
    runs = df[["run", "model", "strategy"]].drop_duplicates()
    for _, x in runs.iterrows():
        same = runs[runs.model == x.model].set_index("strategy")["run"]
        for label, ref in (("vs baseline", BASELINE), ("vs parent", PARENT.get(x.strategy))):
            if not ref or ref == x.strategy or ref not in same.index or (label == "vs parent" and ref == BASELINE):
                continue
            for s in SETS:
                st = paired(preds, same[ref], x.run, s)
                if st and st["n"]:
                    rows.append({"Model": x.model, "Change": f"{ref} → {x.strategy}", "Kind": label, "Set": s,
                                 "Gained": st["gained"], "Lost": st["lost"], "Net": f"{st['net']:+d}",
                                 "McNemar p": f"{st['p_value']:.3f}"})
    out.append("## Paired tests (same questions)\n\n" + md(rows, ["Model", "Change", "Kind", "Set", "Gained",
                                                                 "Lost", "Net", "McNemar p"]))
    # 3. errors on the test sets
    rows = []
    for _, r in df[df.set.isin(["bird_test", "spider_test"])].iterrows():
        errs = r.get("errors") or {}
        top = ", ".join(f"{k} {v}" for k, v in sorted(errs.items(), key=lambda kv: -kv[1]) if k != "correct")[:90]
        rows.append({"Run": r.run, "Set": r.set, "EX": pct(r.ex), "Valid SQL": pct(r.executable_rate),
                     "Simple": pct(r.get("ex_simple")), "Moderate": pct(r.get("ex_moderate")),
                     "Challenging": pct(r.get("ex_challenging")), "Errors": top})
    out.append("## Test-set breakdown\n\n" + md(rows, ["Run", "Set", "EX", "Valid SQL", "Simple", "Moderate",
                                                       "Challenging", "Errors"]))
    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="smollm2", help="comma list of model-name prefixes to include")
    ap.add_argument("--include-smoke", action="store_true", help="also include --limit runs (checking only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    df, preds = load(a.prefix.split(","), a.include_smoke)
    if df.empty:
        sys.exit("no SmolLM2 runs in runs/bench/")
    out.mkdir(parents=True, exist_ok=True)
    cols = TEAM_COLS + [c for c in EXTRA_COLS if c in df]
    df.reindex(columns=cols).to_csv(out / "bench_results.csv", index=False)
    (out / "tables.md").write_text(tables(df, preds), encoding="utf-8")
    for run in preds:
        dst = out / "runs_bench" / run
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("metrics.json", "preds.jsonl", "meta.json"):
            if (ROOT / "runs/bench" / run / f).exists():
                shutil.copy(ROOT / "runs/bench" / run / f, dst / f)
    print(f"{len(preds)} runs -> {out}")


if __name__ == "__main__":
    main()
