"""Metrics beyond the repo's EX.

- EX       : execution accuracy, set(pred rows) == set(gold rows) (repo + BIRD definition).
- Soft-F1  : BIRD mini-dev's partial-credit metric (row-aligned cell overlap), in [0, 1].
- R-VES    : BIRD mini-dev's reward-based valid efficiency score (time ratio gold/pred
             bucketed into rewards 1.25 / 1 / 0.75 / 0.5 / 0.25, 0 if wrong); the run summary
             reports mean(sqrt(reward)) x 100 like the official evaluation_ves.py.
- error_type : coarse failure category for error analysis.
Aggregates: bootstrap 95% CI for EX and paired McNemar vs a reference run.
"""
from __future__ import annotations

import re
import statistics
import time
from functools import lru_cache

import numpy as np

from skillevo.monitor import bootstrap_ci, mcnemar  # noqa: F401  (re-exported)
from skillevo.sqlexec import execute


@lru_cache(maxsize=4096)
def gold_rows(db_path: str, gold_sql: str):
    rows, err = execute(db_path, gold_sql, timeout_s=120)
    return None if err else tuple(rows)


def _row_match(pred_row, gt_row):
    total = len(gt_row) or 1
    matches = sum(1 for v in pred_row if v in gt_row)
    pred_only = len(pred_row) - matches
    truth_only = sum(1 for v in gt_row if v not in pred_row)
    return matches / total, pred_only / total, truth_only / total


def soft_f1(pred_rows, gold) -> float:
    """Port of BIRD mini-dev `calculate_f1_score` (evaluation_f1.py)."""
    if not pred_rows and not gold:
        return 1.0
    pred = list(dict.fromkeys(pred_rows or []))
    gt = list(dict.fromkeys(gold))
    m, p, t = [], [], []
    for i, g in enumerate(gt):
        if i >= len(pred):
            m.append(0); t.append(1)  # noqa: E702
            continue
        a, b, c = _row_match(pred[i], g)
        m.append(a); p.append(b); t.append(c)  # noqa: E702
    for _ in range(len(pred) - len(gt)):
        m.append(0); p.append(1); t.append(0)  # noqa: E702
    tp, fp, fn = sum(m), sum(p), sum(t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _time(db_path, sql, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _, err = execute(db_path, sql, timeout_s=30)
        if err:
            return None
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def rves_reward(db_path: str, pred_sql: str, gold_sql: str, correct: bool, reps: int = 5) -> float:
    if not correct:
        return 0.0
    tp, tg = _time(db_path, pred_sql, reps), _time(db_path, gold_sql, reps)
    if not tp or tg is None:
        return 1.0
    ratio = tg / tp
    return 1.25 if ratio >= 2 else 1.0 if ratio >= 1 else 0.75 if ratio >= 0.5 else 0.5 if ratio >= 0.25 else 0.25


def error_type(err: str | None, correct: bool) -> str:
    if correct:
        return "correct"
    if not err:
        return "wrong_result"
    e = err.lower()
    for pat, lab in ((r"no such column", "bad_column"), (r"no such table", "bad_table"),
                     (r"syntax error|incomplete input|unrecognized token", "syntax"),
                     (r"ambiguous column", "ambiguous_column"), (r"timed out", "timeout"),
                     (r"misuse of aggregate|aggregate", "aggregate_misuse"), (r"no such function", "bad_function")):
        if re.search(pat, e):
            return lab
    return "other_error"


def ex_summary(correct: list[bool], seed: int = 0) -> dict:
    x = np.asarray(correct, dtype=float)
    lo, hi = bootstrap_ci(x, seed=seed)
    return {"n": int(len(x)), "ex": float(x.mean()) if len(x) else 0.0, "ex_ci_lo": lo, "ex_ci_hi": hi}
