"""Tables over finished runs, paired tests, and the head-to-head with Venkat's Qwen results.

    python -m phibench.compare                              # all runs -> results/phi4mini_results.csv
    python -m phibench.compare --pair RUN_A,RUN_B           # McNemar per shared set (A -> B)
    python -m phibench.compare --venkat                     # + his Qwen numbers, same sets

Venkat's numbers are read from his branch with `git show` (read-only; nothing is checked out).
Only his aggregates are in git, so the head-to-head is unpaired; paired tests are ours vs ours.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phibench.metrics import mcnemar  # noqa: E402

RUNS = ROOT / "runs/phibench"
VENKAT_REF = "origin/venkat/qwen-benchmark:results/bench_results.csv"
# our rung -> his closest rung (P4m/P5m used his own rules wording, so these are equivalents, not copies)
EQUIV = {"a1_repo": "zs_repo", "a2_mschema": "zs_mschema", "a3_rules": "ms_rules", "a4_fs3": "ms_fs3"}
COLS = ["ex", "ex_ci_lo", "ex_ci_hi", "ex_first", "soft_f1", "rves", "executable_rate", "repair_rate",
        "avg_llm_seconds", "avg_out_tokens"]


def load_runs(runs_dir: Path = RUNS) -> pd.DataFrame:
    rows = []
    for mf in sorted(runs_dir.glob("*/metrics.json")):
        d = json.loads(mf.read_text())
        meta = d["meta"]
        if meta.get("limit"):  # smoke / partial runs are not results
            continue
        for s, m in d["metrics"].items():
            if m:
                rows.append({"run": mf.parent.name, "label": meta["label"], "strategy": meta["strategy"]["name"],
                             "set": s, **{c: m.get(c) for c in COLS}, "errors": json.dumps(m.get("errors", {}))})
    return pd.DataFrame(rows)


def preds(run: str, runs_dir: Path = RUNS) -> dict:
    out = {}
    for line in open(runs_dir / run / "preds.jsonl"):
        r = json.loads(line)
        out[(r["set"], r["task_id"])] = r["correct"]
    return out


def pair(a: str, b: str, runs_dir: Path = RUNS) -> list[dict]:
    pa, pb = preds(a, runs_dir), preds(b, runs_dir)
    res = []
    for s in sorted({k[0] for k in pa} & {k[0] for k in pb}):
        keys = sorted(k for k in pa if k[0] == s and k in pb)
        res.append({"set": s, **mcnemar([pa[k] for k in keys], [pb[k] for k in keys])})
    return res


def venkat_results() -> pd.DataFrame:
    try:
        raw = subprocess.check_output(["git", "show", VENKAT_REF], cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw))
    return df[df["library"].isna()][["model", "strategy", "rung", "set", "ex", "ex_ci_lo", "ex_ci_hi", "soft_f1", "rves"]]


def head_to_head(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    if ours.empty or theirs.empty:
        return pd.DataFrame()
    rows = []
    for _, r in ours.iterrows():
        eq = EQUIV.get(r["strategy"])
        if not eq:
            continue
        for _, v in theirs[(theirs.strategy == eq) & (theirs.set == r["set"])].iterrows():
            rows.append({"set": r["set"], "ours": f"{r['label']} {r['strategy']}", "our_ex": r["ex"],
                         "theirs": f"{v['model']} {v['strategy']}", "their_ex": v["ex"], "diff_pts": 100 * (r["ex"] - v["ex"])})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="RUN_A,RUN_B")
    ap.add_argument("--venkat", action="store_true")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    if a.pair:
        ra, rb = a.pair.split(",")
        print(pd.DataFrame(pair(ra, rb)).to_string(index=False))
        return
    df = load_runs()
    if df.empty:
        print(f"no finished runs in {RUNS}")
    else:
        print(df.drop(columns=["errors"]).to_string(index=False))
        out = ROOT / "results/phi4mini_results.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n-> {out.relative_to(ROOT)}")
    if a.venkat:
        v = venkat_results()
        if v.empty:
            print("\nVenkat's results not found (fetch origin first)")
            return
        h = head_to_head(df, v)
        print("\nHead-to-head with Venkat's Qwen runs (unpaired; his closest equivalent rung):")
        print(h.to_string(index=False) if not h.empty else "(no matching rungs yet)")
        print("\nHis best per model (P5m, the few-shot rung):")
        print(v[v.strategy == "ms_fs3"].to_string(index=False))


if __name__ == "__main__":
    main()
