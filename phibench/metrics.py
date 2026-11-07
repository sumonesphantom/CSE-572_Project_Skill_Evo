"""Scoring. EX matches the repo (set of result rows equal to the gold rows); the rest follow
BIRD mini-dev's definitions, re-implemented here.

- ex / bootstrap CI : execution accuracy with a 95% percentile bootstrap interval
- soft_f1           : BIRD mini-dev partial credit (row-aligned cell overlap)
- rves              : reward-based valid efficiency score, x100, from timed reruns (median)
- error_type        : one label per prediction, for error analysis
- mcnemar           : exact paired test between two runs on the same questions
"""
from __future__ import annotations

import statistics
import time
from functools import lru_cache

import numpy as np
from scipy.stats import binomtest

from skillevo.sqlexec import execute


@lru_cache(maxsize=8192)
def gold_rows(db_path: str, gold_sql: str):
    rows, err = execute(db_path, gold_sql, timeout_s=180)
    return None if err else tuple(rows)


def bootstrap_ci(x, iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0
    means = np.random.default_rng(seed).choice(x, size=(iters, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def ex_summary(correct: list[bool]) -> dict:
    lo, hi = bootstrap_ci([float(c) for c in correct])
    return {"n": len(correct), "ex": sum(correct) / len(correct) if correct else 0.0, "ex_ci_lo": lo, "ex_ci_hi": hi}


def _row_match(pred_row, gold_row):
    n = len(gold_row) or 1
    matches = sum(1 for v in pred_row if v in gold_row)
    pred_only = sum(1 for v in pred_row if v not in gold_row)
    gold_only = sum(1 for v in gold_row if v not in pred_row)
    return matches / n, pred_only / n, gold_only / n


def soft_f1(pred, gold) -> float:
    pred = list(dict.fromkeys(tuple(r) for r in (pred or [])))
    gold = list(dict.fromkeys(tuple(r) for r in (gold or [])))
    if not pred and not gold:
        return 1.0
    tp = fp = fn = 0.0
    for i, g in enumerate(gold):
        if i >= len(pred):
            fn += 1
            continue
        m, p_only, g_only = _row_match(pred[i], g)
        tp, fp, fn = tp + m, fp + p_only, fn + g_only
    fp += max(len(pred) - len(gold), 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _median_time(db_path: str, sql: str, runs: int, timeout_s: float) -> float | None:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _, err = execute(db_path, sql, timeout_s)
        if err:
            return None
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def rves_reward(db_path: str, pred_sql: str, gold_sql: str, correct: bool, runs: int = 5) -> float:
    if not correct:
        return 0.0
    tp = _median_time(db_path, pred_sql, runs, 60)
    tg = _median_time(db_path, gold_sql, runs, 60)
    if not tp or not tg:
        return 1.0
    tau = tg / tp
    return 1.25 if tau >= 2 else 1.0 if tau >= 1 else 0.75 if tau >= 0.5 else 0.5 if tau >= 0.25 else 0.25


def error_type(err: str | None, correct: bool, n_rows: int = -1) -> str:
    if correct:
        return "correct"
    if err is None:
        return "empty_result" if n_rows == 0 else "wrong_result"
    e = err.lower()
    if "no such column" in e or "ambiguous column" in e:
        return "bad_column"
    if "no such table" in e:
        return "bad_table"
    if "no such function" in e:
        return "bad_function"
    if "syntax error" in e or "incomplete input" in e or "unrecognized token" in e:
        return "syntax"
    if "timed out" in e:
        return "timeout"
    return "other_error"


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired change from run a to run b on the same questions."""
    assert len(a) == len(b)
    gained = sum(1 for x, y in zip(a, b) if not x and y)
    lost = sum(1 for x, y in zip(a, b) if x and not y)
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {"n": len(a), "gained": gained, "lost": lost, "net": gained - lost, "p_value": float(p)}
