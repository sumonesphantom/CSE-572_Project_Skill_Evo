"""Aggregate every benchmark / SkillEvo run into the Excel tracker and the markdown report.

    python -m bench.report            # -> results/qwen_benchmark.xlsx, results/bench_results.csv,
                                      #    and refreshes the AUTO blocks in docs/QWEN_BENCHMARK.md
Runs made with --limit (smoke tests) are skipped.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.plan import MODELS, RUNGS, SETS, plan  # noqa: E402
from bench.strategies import LADDER, RUNG  # noqa: E402
from skillevo.monitor import mcnemar  # noqa: E402

LADDER_ORDER = RUNGS[:2] + ["zs_repo_full"] + RUNGS[2:] + ["ms_rules", "ms_fs3", "ms_fs3_sc5"]
# the rung each strategy changes exactly one thing from (first one present is used)
PARENT = {"zs_repo": ["zs_ddl"], "zs_repo_full": ["zs_repo"], "zs_mschema": ["zs_repo_full", "zs_repo"],
          "zs_desc": ["zs_mschema"], "zs_rules": ["zs_desc"], "fs3": ["zs_rules"], "fs3_cot": ["fs3"],
          "fs3_sc5": ["fs3"], "fs3_cot_sc5": ["fs3_cot"], "ms_rules": ["zs_mschema"], "ms_fs3": ["ms_rules"],
          "ms_fs3_sc5": ["ms_fs3"]}


def parent(st, runs):
    return next((p for p in PARENT.get(st, []) if p in runs), None)


OUT_X = ROOT / "results/qwen_benchmark.xlsx"
OUT_CSV = ROOT / "results/bench_results.csv"
DOC = ROOT / "docs/QWEN_BENCHMARK.md"
SETS_L = SETS.split(",")

FONT = "Arial"
F_BASE = Font(name=FONT, size=10)
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_INPUT = Font(name=FONT, size=10, color="0000FF")
F_TITLE = Font(name=FONT, size=14, bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color="555555")
FILL_HEAD = PatternFill("solid", fgColor="1F3A5F")
FILL_INPUT = PatternFill("solid", fgColor="FFF7CC")
FILL_BEST = PatternFill("solid", fgColor="D9EAD3")
THIN = Border(bottom=Side(style="thin", color="DDDDDD"))
PCT = "0.0%"


# ---------------------------------------------------------------------- loading
def load_bench():
    rows, preds = [], {}
    for d in sorted((ROOT / "runs/bench").glob("*")):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        j = json.loads(mf.read_text())
        meta = j["meta"]
        if meta.get("limit"):
            continue
        p = defaultdict(dict)
        for line in open(d / "preds.jsonl"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p[r["set"]][r["task_id"]] = r
        preds[d.name] = p
        for s, m in j["metrics"].items():
            if not m:
                continue
            rows.append({"run": d.name, "model": meta["model"], "strategy": meta["strategy"]["name"],
                         "rung": meta.get("rung", RUNG.get(meta["strategy"]["name"], "")),
                         "library": Path(meta["library"]).parent.parent.name if meta.get("library") else "",
                         "set": s, **{k: v for k, v in m.items() if k != "errors"},
                         "errors": m.get("errors", {})})
    return pd.DataFrame(rows), preds


def paired(preds, run_a, run_b, s):
    a, b = preds.get(run_a, {}).get(s, {}), preds.get(run_b, {}).get(s, {})
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return mcnemar([a[i]["correct"] for i in ids], [b[i]["correct"] for i in ids])


def load_skillevo():
    from skillevo.analysis import Run, list_runs, summarize
    out = []
    for p in list_runs(ROOT / "runs"):
        try:
            s = summarize(Run(p))
        except Exception as e:  # noqa: BLE001 - a half-written run should not break the report
            s = {"run": p.name, "error": str(e)}
        cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
        s["solver"] = cfg.get("llm", {}).get("solver")
        s["evolver"] = cfg.get("llm", {}).get("evolver")
        out.append(s)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- sheet helpers
def header(ws, row, cols, widths=None):
    for c, name in enumerate(cols, 1):
        cell = ws.cell(row=row, column=c, value=name)
        cell.font, cell.fill = F_HEAD, FILL_HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[row].height = 30
    if widths:
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def put(ws, row, col, val, fmt=None, font=F_BASE, fill=None):
    cell = ws.cell(row=row, column=col, value=val)
    cell.font = font
    if fmt:
        cell.number_format = fmt
    if fill:
        cell.fill = fill
    cell.border = THIN
    return cell


def title(ws, text, sub=None):
    ws["A1"] = text
    ws["A1"].font = F_TITLE
    if sub:
        ws["A2"] = sub
        ws["A2"].font = F_NOTE


# ---------------------------------------------------------------------- workbook
RES_COLS = ["run", "model", "strategy", "rung", "library", "set", "n", "ex", "ex_ci_lo", "ex_ci_hi", "ΔEX vs zs_repo",
            "ex_first_attempt", "soft_f1", "rves", "executable_rate", "repair_rate", "oracle_ex",
            "ex_simple", "ex_moderate", "ex_challenging", "avg_in_tokens", "avg_out_tokens", "avg_llm_seconds", "avg_llm_calls"]
PCT_COLS = {"ex", "ex_ci_lo", "ex_ci_hi", "ΔEX vs zs_repo", "ex_first_attempt", "soft_f1", "executable_rate",
            "repair_rate", "oracle_ex", "ex_simple", "ex_moderate", "ex_challenging"}


def sheet_readme(wb):
    ws = wb.active
    ws.title = "How_to_use"
    title(ws, "Qwen Text-to-SQL benchmark tracker (CSE 572 SkillEvo)",
          "Generated by `python -m bench.report`. Re-running rewrites the whole file, so copy anything typed into blue cells somewhere else first.")
    lines = [
        ("Sheets", ""),
        ("Experiment_Plan", "Every planned run (from bench/plan.py): question it answers, command, est. hours, status read from disk. Blue columns = your priority/notes."),
        ("Results", "One row per (run, eval set). EX is the repo's primary metric; ΔEX vs zs_repo is a live formula against the same model's repo-prompt row."),
        ("Prompt_Ladder", "P0..P8 rungs for Qwen2.5-Coder-7B, EX per set (formulas into Results) + chart."),
        ("Model_Sweep", "Key rungs x the four Qwen models on bird_test / spider_test."),
        ("Paired_Tests", "McNemar on identical tasks: each rung vs previous rung and vs zs_repo (gained / lost / net / p)."),
        ("SkillEvo", "Self-evolving skill arms: start/final/gain, drift, kept-negative edits, audit agreement."),
        ("FineTune_Grid", "LoRA hyper-parameter grid. Blue cells are inputs: try a row, then type its EX in."),
        ("Prompt_Catalog", "What each prompt rung changes."),
        ("Errors", "Failure categories per run and set."),
        ("Per_DB", "EX per database (BIRD) per run: where each method helps or hurts."),
        ("", ""),
        ("Colour legend", ""),
        ("Blue text on yellow", "input cell (yours to edit)"),
        ("Black", "computed value or formula"),
        ("Green fill", "best value in its column/group"),
        ("", ""),
        ("Metrics", ""),
        ("EX", "execution accuracy: set(predicted rows) == set(gold rows). Same as skillevo.sqlexec.is_correct and BIRD's EX."),
        ("ex_first_attempt", "EX before the one SQLite-error repair turn (repair effect = ex - ex_first_attempt)."),
        ("soft_f1", "BIRD mini-dev Soft-F1 (partial credit for row/column overlap)."),
        ("rves", "BIRD mini-dev R-VES x100 (efficiency-weighted EX; BIRD sets only; median of 5 timings)."),
        ("oracle_ex", "for self-consistency: share of tasks where ANY candidate was right (upper bound of voting)."),
        ("CI", "bootstrap 95% interval of EX (2,000 resamples)."),
    ]
    for i, (a, b) in enumerate(lines, 4):
        ws.cell(row=i, column=1, value=a).font = Font(name=FONT, size=10, bold=not b)
        ws.cell(row=i, column=2, value=b).font = F_BASE
    ws["A17"].font, ws["A17"].fill = F_INPUT, FILL_INPUT
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 120


def sheet_plan(wb):
    ws = wb.create_sheet("Experiment_Plan")
    cols = ["ID", "Family", "Model", "Method", "Question it answers", "Est. hours", "Status", "Needs", "Command(s)",
            "Your priority (1-3)", "Your notes"]
    title(ws, "Experiment plan", "Status is read from runs/ on disk each time the report is built. Priority/notes are inputs (blue).")
    header(ws, 3, cols, [6, 11, 20, 34, 52, 9, 9, 10, 90, 10, 30])
    dv = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
    ws.add_data_validation(dv)
    for r, e in enumerate(plan(), 4):
        vals = [e.id, e.family, e.model, e.method, e.question, e.est_hours, e.status(), ",".join(e.needs), "\n".join(e.cmds)]
        for c, v in enumerate(vals, 1):
            cell = put(ws, r, c, v, "0.0" if c == 6 else None)
            if c in (5, 9):
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        if e.status() == "done":
            ws.cell(row=r, column=7).fill = FILL_BEST
        for c in (10, 11):
            put(ws, r, c, None, font=F_INPUT, fill=FILL_INPUT)
        dv.add(ws.cell(row=r, column=10))
    last = 3 + len(plan())
    put(ws, last + 2, 5, "Total estimated hours", font=Font(name=FONT, size=10, bold=True))
    put(ws, last + 2, 6, f"=SUM(F4:F{last})", "0.0")
    put(ws, last + 3, 5, "Hours still to run (status <> done)", font=Font(name=FONT, size=10, bold=True))
    put(ws, last + 3, 6, f'=SUMIFS(F4:F{last},G4:G{last},"<>done")', "0.0")
    ws.auto_filter.ref = f"A3:K{last}"


def sheet_results(wb, df):
    ws = wb.create_sheet("Results")
    title(ws, "Results: one row per run x eval set", "Select prompts on bird_val; report bird_test and spider_test. ΔEX column is a formula.")
    header(ws, 3, RES_COLS, [34, 20, 13, 6, 14, 12, 6] + [9] * (len(RES_COLS) - 7))
    if df.empty:
        put(ws, 4, 1, "no finished runs yet")
        return
    d = df.sort_values(["model", "set", "rung"]).reset_index(drop=True)
    n = len(d)
    last = 3 + n
    col = {c: i for i, c in enumerate(RES_COLS, 1)}
    L = {c: get_column_letter(i) for c, i in col.items()}
    for i, row in d.iterrows():
        r = 4 + i
        for c in RES_COLS:
            if c == "ΔEX vs zs_repo":
                # same model, same set, strategy zs_repo, no library
                base = (f'SUMIFS(${L["ex"]}$4:${L["ex"]}${last},${L["model"]}$4:${L["model"]}${last},{L["model"]}{r},'
                        f'${L["set"]}$4:${L["set"]}${last},{L["set"]}{r},${L["strategy"]}$4:${L["strategy"]}${last},"zs_repo",'
                        f'${L["library"]}$4:${L["library"]}${last},"")')
                cnt = base.replace("SUMIFS(", "COUNTIFS(").replace(f'${L["ex"]}$4:${L["ex"]}${last},', "", 1)
                put(ws, r, col[c], f'=IF({cnt}=0,"",{L["ex"]}{r}-{base})', "+0.0%;-0.0%;0.0%")
                continue
            v = row.get(c)
            if isinstance(v, float) and pd.isna(v):
                v = None
            put(ws, r, col[c], v, PCT if c in PCT_COLS else "0.0" if c in ("rves", "avg_llm_seconds", "avg_llm_calls") else
                "#,##0" if c in ("avg_in_tokens", "avg_out_tokens") else None)
    # highlight best EX per (model, set)
    for (_m, _s), g in d.groupby(["model", "set"]):
        put_fill = g["ex"].idxmax()
        ws.cell(row=4 + put_fill, column=col["ex"]).fill = FILL_BEST
    ws.auto_filter.ref = f"A3:{get_column_letter(len(RES_COLS))}{last}"
    ws.cell(row=last + 2, column=1, value="ΔEX: =ex - SUMIFS(ex, same model, same set, strategy=\"zs_repo\", no library). Blank when that baseline row is missing.").font = F_NOTE


def _lookup(model, strategy, s, n_rows):
    return (f'=IFERROR(INDEX(Results!$H$4:$H${n_rows + 3},MATCH(1,INDEX((Results!$B$4:$B${n_rows + 3}="{model}")*'
            f'(Results!$C$4:$C${n_rows + 3}="{strategy}")*(Results!$F$4:$F${n_rows + 3}="{s}")*(Results!$E$4:$E${n_rows + 3}=""),0),0)),"")')


def sheet_ladder(wb, df):
    ws = wb.create_sheet("Prompt_Ladder")
    m7 = MODELS["coder7b"]
    title(ws, f"Prompt ladder on {m7}", "Each rung adds one change to the previous one. Cells are formulas into Results (blank = not run yet).")
    header(ws, 3, ["Rung", "Strategy", "What it adds"] + SETS_L, [6, 14, 70, 12, 12, 12])
    n = max(len(df), 1)
    adds = catalog()
    for i, r in enumerate(LADDER_ORDER):
        row = 4 + i
        put(ws, row, 1, RUNG[r])
        put(ws, row, 2, r)
        put(ws, row, 3, adds[r])
        for j, s in enumerate(SETS_L):
            put(ws, row, 4 + j, _lookup(m7, r, s, n), PCT)
    ch = BarChart()
    ch.type = "col"
    ch.title = "EX by prompt rung (Qwen2.5-Coder-7B)"
    ch.y_axis.title = "EX"
    ch.y_axis.numFmt = "0%"
    ch.add_data(Reference(ws, min_col=4, max_col=3 + len(SETS_L), min_row=3, max_row=3 + len(LADDER_ORDER)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=2, min_row=4, max_row=3 + len(LADDER_ORDER)))
    ch.height, ch.width = 9, 22
    ws.add_chart(ch, "H3")


def sheet_sweep(wb, df):
    ws = wb.create_sheet("Model_Sweep")
    title(ws, "Model sweep: EX on held-out sets", "Formulas into Results. Compare a rung across models, or the best rung per model.")
    rungs = ["zs_repo", "zs_rules", "fs3", "fs3_cot", "fs3_sc5"]
    n = max(len(df), 1)
    row = 3
    for s in ("bird_test", "spider_test", "bird_val"):
        put(ws, row, 1, s, font=Font(name=FONT, size=11, bold=True))
        row += 1
        header(ws, row, ["Model"] + rungs + ["best"], [24] + [11] * (len(rungs) + 1))
        ws.freeze_panes = None
        for k, (key, m) in enumerate(MODELS.items()):
            r = row + 1 + k
            put(ws, r, 1, m)
            for j, st in enumerate(rungs):
                put(ws, r, 2 + j, _lookup(m, st, s, n), PCT)
            put(ws, r, 2 + len(rungs), f"=IF(COUNT(B{r}:{get_column_letter(1 + len(rungs))}{r})=0,\"\",MAX(B{r}:{get_column_letter(1 + len(rungs))}{r}))", PCT)
        row += len(MODELS) + 3


def sheet_paired(wb, df, preds):
    ws = wb.create_sheet("Paired_Tests")
    title(ws, "Paired McNemar tests on identical tasks", "Each rung vs the rung it changes one thing from (PARENT in bench/report.py), and vs zs_repo. Computed in Python; net = gained - lost.")
    cols = ["model", "set", "run", "vs", "n", "gained", "lost", "net", "Δ EX", "p (exact McNemar)", "significant @0.05"]
    header(ws, 3, cols, [20, 12, 34, 34, 6, 8, 8, 8, 9, 12, 12])
    r = 4
    if df.empty:
        return
    for (model, s), g in df[df.library == ""].groupby(["model", "set"]):
        runs = {row.strategy: row.run for row in g.itertuples()}
        order = [x for x in LADDER_ORDER if x in runs]
        for st in order:
            refs = []
            par = parent(st, runs)
            if par:
                refs.append(runs[par])
            if st != "zs_repo" and "zs_repo" in runs and par != "zs_repo":
                refs.append(runs["zs_repo"])
            for ref in refs:
                t = paired(preds, ref, runs[st], s)
                if not t:
                    continue
                vals = [model, s, runs[st], ref, t["n"], t["gained"], t["lost"], t["net"], t["delta"], t["p_value"], t["p_value"] < 0.05]
                for c, v in enumerate(vals, 1):
                    put(ws, r, c, v, "+0.0%;-0.0%;0.0%" if c == 9 else "0.000" if c == 10 else None,
                        fill=FILL_BEST if c == 11 and v and t["net"] > 0 else None)
                r += 1
    # library runs vs same model+strategy without library
    for row in df[df.library != ""].itertuples():
        base = df[(df.model == row.model) & (df.strategy == row.strategy) & (df.library == "") & (df.set == row.set)]
        if base.empty:
            continue
        t = paired(preds, base.run.iloc[0], row.run, row.set)
        if t:
            vals = [row.model, row.set, row.run, base.run.iloc[0], t["n"], t["gained"], t["lost"], t["net"], t["delta"], t["p_value"], t["p_value"] < 0.05]
            for c, v in enumerate(vals, 1):
                put(ws, r, c, v, "+0.0%;-0.0%;0.0%" if c == 9 else "0.000" if c == 10 else None)
            r += 1


SE_COLS = ["run", "arm", "solver", "evolver", "rounds", "val_start", "val_final", "val_best", "bird_test_start", "bird_test_final",
           "bird_test_gain", "spider_transfer_start", "spider_transfer_final", "val_max_drawdown", "val_forgetting_rate",
           "library_final", "edits", "edits_kept", "edits_positive", "edits_negative", "kept_negative", "net_drift_kept",
           "audit_net_kept", "audit_sign_agreement", "audit_kappa", "rollbacks", "skills_helpful", "skills_harmful"]


def sheet_skillevo(wb, se):
    ws = wb.create_sheet("SkillEvo")
    title(ws, "Self-evolving skills (SkillEvo arms)", "From skillevo.analysis.summarize. Positive gain + low kept_negative + high audit agreement = healthy evolution.")
    header(ws, 3, SE_COLS, [32, 11, 18, 18] + [9] * (len(SE_COLS) - 4))
    if se.empty:
        put(ws, 4, 1, "no SkillEvo runs yet (see Experiment_Plan phase C)")
        return
    for i, row in se.reset_index(drop=True).iterrows():
        for c, k in enumerate(SE_COLS, 1):
            v = row.get(k)
            if isinstance(v, float) and pd.isna(v):
                v = None
            put(ws, 4 + i, c, v, PCT if any(x in k for x in ("_start", "_final", "_best", "_gain", "drawdown", "rate")) and "library" not in k else
                "0.00" if k in ("audit_sign_agreement", "audit_kappa") else None)


FT_COLS = ["ID", "Base model", "Rank", "Scale", "Layers", "LR", "Iters", "Eff. batch", "Train data", "Prompt", "mask_prompt",
           "Why try it", "Status", "EX bird_val", "EX bird_test", "EX spider_test", "Notes"]


def ft_grid():
    rows = []
    for key in ("coder1.5b", "coder3b", "coder7b"):
        it = {"coder1.5b": 1200, "coder3b": 1000, "coder7b": 800}[key]
        rows.append([f"FT-{key}-r16", key, 16, 2.0, 16, 1e-4, it, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "default (D*1 in plan)"])
    rows += [
        ["FT-coder3b-r8", "coder3b", 8, 2.0, 16, 1e-4, 1000, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "lower capacity: does r8 suffice?"],
        ["FT-coder3b-r32", "coder3b", 32, 2.0, 16, 1e-4, 1000, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "more capacity"],
        ["FT-coder3b-all", "coder3b", 16, 2.0, 36, 1e-4, 1000, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "LoRA on all layers"],
        ["FT-coder3b-lr2e4", "coder3b", 16, 2.0, 16, 2e-4, 1000, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "faster LR"],
        ["FT-coder3b-spider7k", "coder3b", 16, 2.0, 16, 1e-4, 2000, 8, "Spider 7k + BIRD-train x3", "ms_rules", True, "full Spider train"],
        ["FT-coder3b-birdx10", "coder3b", 16, 2.0, 16, 1e-4, 1000, 8, "Spider 3k + BIRD-train x10", "ms_rules", True, "BIRD-heavy mix (risk: overfit 247 q's)"],
        ["FT-coder3b-nomask", "coder3b", 16, 2.0, 16, 1e-4, 1000, 8, "Spider 3k + BIRD-train x3", "ms_rules", False, "loss on prompt too (ablation)"],
        ["FT-coder3b-repo", "coder3b", 16, 2.0, 16, 1e-4, 1000, 8, "Spider 3k + BIRD-train x3", "zs_repo", True, "train/eval on the repo prompt"],
        ["FT-coder7b-dora", "coder7b", 16, 2.0, 16, 1e-4, 800, 8, "Spider 3k + BIRD-train x3", "ms_rules", True, "fine_tune_type: dora"],
    ]
    return rows


def sheet_ft(wb, df):
    ws = wb.create_sheet("FineTune_Grid")
    title(ws, "LoRA fine-tuning grid (MLX-LM)", "Blue cells are inputs. Results columns auto-fill from Results when the adapter was evaluated as mlx:<key>-r16; otherwise type them in.")
    header(ws, 3, FT_COLS, [20, 11, 6, 6, 7, 8, 7, 8, 26, 10, 10, 34, 10, 10, 10, 10, 30])
    n = max(len(df), 1)
    for i, row in enumerate(ft_grid()):
        r = 4 + i
        for c, v in enumerate(row, 1):
            put(ws, r, c, v, "0.0E+00" if c == 6 else None, font=F_INPUT, fill=FILL_INPUT)
        label = "mlx:" + row[0][3:]  # FT-coder3b-r16 -> mlx:coder3b-r16
        evaluated = not df.empty and (df.model == label).any()
        put(ws, r, 13, "done" if evaluated else "planned", font=F_INPUT, fill=FILL_INPUT)
        for j, s in enumerate(SETS_L):
            if evaluated:
                put(ws, r, 14 + j, _lookup(label, "ms_rules", s, n), PCT)
            else:
                put(ws, r, 14 + j, None, PCT, font=F_INPUT, fill=FILL_INPUT)
        put(ws, r, 17, None, font=F_INPUT, fill=FILL_INPUT)
    r = 5 + len(ft_grid())
    ws.cell(row=r, column=1, value="Example row format: FT-coder3b-r16 | coder3b | 16 | 2.0 | 16 | 1.0E-04 | 1000 | 8 | Spider 3k + BIRD-train x3 | zs_rules | TRUE | why | done | 41.8% ...").font = F_NOTE
    ws.cell(row=r + 1, column=1, value="Edit finetune/configs/<key>_r16.yaml (rank -> lora_parameters.rank, Layers -> num_layers, LR -> learning_rate) and change adapter_path to match the ID.").font = F_NOTE


def catalog():
    return {
        "zs_ddl": "raw CREATE TABLE statements, nothing else",
        "zs_repo": "repo default: DDL + 2 sample rows per table (capped at 6,000 chars, so large BIRD schemas get truncated)",
        "zs_repo_full": "control: repo prompt with the 20,000-char cap instead of 6,000",
        "zs_mschema": "compact typed column list with PK, 3 distinct example values per column, FK lines; 20k-char cap",
        "zs_desc": "+ BIRD column / value descriptions from database_description/*.csv",
        "zs_rules": "+ 5 output rules: exact columns, literal evidence formulas, backtick quoting, CAST for ratios, LIMIT only for top-k",
        "fs3": "+ 3 retrieved solved examples (BM25 on question+evidence, same database first, train split only)",
        "fs3_cot": "+ brief numbered reasoning before the final SQL block",
        "fs3_sc5": "fs3 + self-consistency: greedy + 4 samples at T=0.7, majority vote on execution results",
        "fs3_cot_sc5": "fs3_cot + the same 5-way execution vote",
        "ms_rules": "branch: P2 schema (no descriptions) + the 5 rules",
        "ms_fs3": "branch: ms_rules + 3 retrieved examples",
        "ms_fs3_sc5": "branch: ms_fs3 + 5-way execution vote",
    }


def sheet_catalog(wb):
    ws = wb.create_sheet("Prompt_Catalog")
    title(ws, "Prompt strategies", "Defined in bench/strategies.py. All rungs keep the repo's one-shot repair on SQLite errors.")
    header(ws, 3, ["Rung", "Strategy", "Schema", "Rules", "Shots", "CoT", "Samples", "Max tokens", "What it adds"],
           [6, 14, 14, 7, 7, 6, 8, 10, 100])
    cat = catalog()
    for i, (name, s) in enumerate(LADDER.items(), 4):
        for c, v in enumerate([RUNG[name], name, s.schema, s.rules, s.k_shots, s.cot, s.n_samples, s.max_tokens, cat[name]], 1):
            put(ws, i, c, v)


def sheet_errors(wb, df):
    ws = wb.create_sheet("Errors")
    title(ws, "Failure categories", "wrong_result = query ran but rows differ; others are SQLite errors after the repair turn.")
    cats = ["correct", "wrong_result", "bad_column", "bad_table", "syntax", "ambiguous_column", "aggregate_misuse", "bad_function", "timeout", "other_error"]
    header(ws, 3, ["run", "set", "n"] + cats, [34, 12, 6] + [11] * len(cats))
    for i, row in enumerate(df.itertuples(), 4):
        put(ws, i, 1, row.run)
        put(ws, i, 2, row.set)
        put(ws, i, 3, row.n)
        for j, c in enumerate(cats):
            put(ws, i, 4 + j, row.errors.get(c, 0))


def sheet_perdb(wb, preds):
    ws = wb.create_sheet("Per_DB")
    title(ws, "EX per BIRD database (bird_val + bird_test pooled)", "Spot databases where a method helps or hurts.")
    table = {}
    dbs = set()
    for run, p in preds.items():
        c = defaultdict(list)
        for s in ("bird_val", "bird_test"):
            for r in p.get(s, {}).values():
                c[r["db_id"]].append(r["correct"])
        if c:
            table[run] = {db: sum(v) / len(v) for db, v in c.items()}
            dbs |= set(c)
    dbs = sorted(dbs)
    header(ws, 3, ["run"] + dbs, [34] + [12] * len(dbs))
    for i, (run, row) in enumerate(sorted(table.items()), 4):
        put(ws, i, 1, run)
        for j, db in enumerate(dbs):
            put(ws, i, 2 + j, row.get(db), PCT)


def build_xlsx(df, preds, se):
    wb = Workbook()
    sheet_readme(wb)
    sheet_plan(wb)
    sheet_results(wb, df)
    sheet_ladder(wb, df)
    sheet_sweep(wb, df)
    sheet_paired(wb, df, preds)
    sheet_skillevo(wb, se)
    sheet_ft(wb, df)
    sheet_catalog(wb)
    sheet_errors(wb, df)
    sheet_perdb(wb, preds)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.font is None or cell.font.name != FONT:
                    cell.font = Font(name=FONT, size=cell.font.size if cell.font else 10, bold=cell.font.bold if cell.font else False)
    wb.calculation.fullCalcOnLoad = True  # formulas have no cached values until Excel opens the file
    OUT_X.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_X)


# ---------------------------------------------------------------------- markdown AUTO blocks
def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{100 * x:.1f}"


def md_blocks(df, preds, se):
    blocks = {}
    if df.empty:
        blocks["RESULTS"] = "_No finished benchmark runs yet._"
    else:
        rows = []
        for (model, strat, lib), g in df.groupby(["model", "strategy", "library"], sort=False):
            by = {r.set: r for r in g.itertuples()}
            cell = lambda s, k="ex": pct(getattr(by[s], k)) if s in by else ""  # noqa: E731
            ci = lambda s: f"{cell(s)} ({pct(by[s].ex_ci_lo)}–{pct(by[s].ex_ci_hi)})" if s in by else ""  # noqa: E731
            rv = f"{by['bird_test'].rves:.1f}" if "bird_test" in by and not pd.isna(getattr(by["bird_test"], "rves", float("nan"))) else ""
            rows.append([model, RUNG.get(strat, ""), strat, lib or "–", cell("bird_val"), ci("bird_test"), cell("bird_test", "soft_f1"), rv, ci("spider_test")])
        order = {m: i for i, m in enumerate(MODELS.values())}
        rows.sort(key=lambda r: (order.get(r[0], 99), r[1], r[3]))
        blocks["RESULTS"] = md_table(rows, ["Model", "Rung", "Strategy", "Skills", "BIRD val EX", "BIRD test EX (95% CI)", "BIRD test Soft-F1", "BIRD test R-VES", "Spider test EX (95% CI)"])
        tests = []
        m7 = MODELS["coder7b"]
        g = df[(df.model == m7) & (df.library == "")]
        runs = {r.strategy: r.run for r in g.itertuples()}
        order_r = [x for x in LADDER_ORDER if x in runs]
        for s in SETS_L:
            for st in order_r:
                par = parent(st, runs)
                t = paired(preds, runs[par], runs[st], s) if par else None
                if t:
                    tests.append([s, f"{par} → {st}", t["gained"], t["lost"], f"{t['net']:+d}", f"{t['p_value']:.3f}"])
        blocks["PAIRED"] = md_table(tests, ["Set", "Change", "Gained", "Lost", "Net", "McNemar p"]) if tests else "_Needs at least two rungs on the same set._"
    if se.empty:
        blocks["SKILLEVO"] = "_No SkillEvo runs yet._"
    else:
        rows = [[r.get("run"), r.get("evolver"), pct(r.get("bird_test_start")), pct(r.get("bird_test_final")), pct(r.get("spider_transfer_final")),
                 r.get("edits", ""), r.get("edits_kept", ""), r.get("kept_negative", ""), r.get("library_final", "")] for _, r in se.iterrows()]
        blocks["SKILLEVO"] = md_table(rows, ["Run", "Evolver", "BIRD test start", "BIRD test final", "Spider final", "Edits", "Kept", "Kept negative", "Library size"])
    blocks["PLAN"] = md_table([[e.id, e.family, e.model, e.method, e.status(), f"{e.est_hours:.1f}"] for e in plan()],
                              ["ID", "Family", "Model", "Method", "Status", "Est. h"])
    return blocks


def update_md(blocks):
    if not DOC.exists():
        return
    text = DOC.read_text()
    for k, v in blocks.items():
        text = re.sub(rf"(<!-- AUTO:{k} -->)(.*?)(<!-- /AUTO:{k} -->)", lambda m: f"{m.group(1)}\n{v}\n{m.group(3)}", text, flags=re.S)
    DOC.write_text(text)


def main():
    df, preds = load_bench()
    se = load_skillevo()
    if not df.empty:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["errors"]).to_csv(OUT_CSV, index=False)
    build_xlsx(df, preds, se)
    update_md(md_blocks(df, preds, se))
    print(f"{len(df)} result rows, {len(se)} SkillEvo runs -> {OUT_X.relative_to(ROOT)}, {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
