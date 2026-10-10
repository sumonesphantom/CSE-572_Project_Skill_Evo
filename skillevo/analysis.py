"""Load run logs and compute evolution-quality metrics (used by the dashboard and analyze.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("positive", "neutral", "negative", "inert")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


class Run:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.name = self.dir.name
        cfg = self.dir / "config.json"
        self.config = json.loads(cfg.read_text()) if cfg.exists() else {}
        self.arm = self.config.get("arm", "?")
        self.rounds = read_jsonl(self.dir / "rounds.jsonl")
        self.edits = read_jsonl(self.dir / "edits.jsonl")
        self.versions = read_jsonl(self.dir / "skill_versions.jsonl")
        self.stats = read_jsonl(self.dir / "skill_stats.jsonl")
        self.events = read_jsonl(self.dir / "events.jsonl")
        self._traj = None

    @property
    def trajectories(self) -> pd.DataFrame:
        if self._traj is None:
            self._traj = read_jsonl(self.dir / "trajectories.jsonl")
        return self._traj

    def accuracy(self) -> pd.DataFrame:
        """Long table: round, split, accuracy."""
        if self.rounds.empty:
            return pd.DataFrame(columns=["round", "split", "accuracy"])
        rows = [{"round": r["round"], "split": k, "accuracy": v} for _, r in self.rounds.iterrows() for k, v in r["acc"].items()]
        return pd.DataFrame(rows)

    def library(self, round_: int) -> dict:
        p = self.dir / "library" / f"round_{round_:02d}.json"
        return json.loads(p.read_text()) if p.exists() else {"skills": []}


def max_drawdown(series) -> float:
    peak, dd = -np.inf, 0.0
    for v in series:
        peak = max(peak, v)
        dd = max(dd, peak - v)
    return float(dd)


def forgetting(traj: pd.DataFrame, phase: str = "val") -> dict:
    """Of the tasks solved at some round, how many are unsolved at the final round?"""
    if traj.empty:
        return {"ever_solved": 0, "forgotten": 0, "forgetting_rate": 0.0}
    t = traj[traj.phase == phase]
    last = t["round"].max()
    piv = t.pivot_table(index="task_id", columns="round", values="correct", aggfunc="last")
    ever = piv.any(axis=1)
    final = piv[last].fillna(False).astype(bool)
    forgotten = int((ever & ~final).sum())
    return {"ever_solved": int(ever.sum()), "forgotten": forgotten,
            "forgetting_rate": forgotten / max(int(ever.sum()), 1)}


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def monitor_audit(edits: pd.DataFrame) -> dict:
    """Does the validation verdict agree with the held-out audit verdict?"""
    if edits.empty or "a_label" not in edits:
        return {}
    e = edits[(edits.v_label != "inert") & (edits.a_label != "inert")]
    if e.empty:
        return {"n_audited": 0}
    sign = lambda x: np.sign(x)  # noqa: E731
    pos = e[e.v_label == "positive"]
    neg = e[e.v_label == "negative"]
    return {
        "n_audited": len(e),
        "label_agreement": float((e.v_label == e.a_label).mean()),
        "sign_agreement": float((sign(e.v_net) == sign(e.a_net)).mean()),
        "kappa": cohen_kappa(e.v_label, e.a_label),
        "precision_positive": float((pos.a_net > 0).mean()) if len(pos) else float("nan"),
        "precision_negative": float((neg.a_net < 0).mean()) if len(neg) else float("nan"),
    }


def summarize(run: Run) -> dict:
    acc = run.accuracy()
    out = {"run": run.name, "arm": run.arm, "rounds": int(run.rounds["round"].max()) if not run.rounds.empty else 0}
    for split, g in acc.groupby("split"):
        g = g.sort_values("round")
        out[f"{split}_start"] = g.accuracy.iloc[0]
        out[f"{split}_final"] = g.accuracy.iloc[-1]
        out[f"{split}_best"] = g.accuracy.max()
        out[f"{split}_gain"] = g.accuracy.iloc[-1] - g.accuracy.iloc[0]
        out[f"{split}_aulc"] = g.accuracy.mean()
        out[f"{split}_max_drawdown"] = max_drawdown(g.accuracy)
    if not run.rounds.empty:
        out["library_final"] = int(run.rounds.library_size.iloc[-1])
        out["rollbacks"] = int(run.rounds.rolled_back.sum())
    e = run.edits
    if not e.empty:
        out["edits"] = len(e)
        out["edits_kept"] = int(e.accepted.sum())
        for lbl in LABELS:
            out[f"edits_{lbl}"] = int((e.v_label == lbl).sum())
        kept = e[e.accepted]
        out["kept_negative"] = int((kept.v_label == "negative").sum())
        out["harmful_kept_rate"] = out["kept_negative"] / max(len(kept), 1)
        out["net_drift_kept"] = int(kept.v_net.sum())
        if "a_net" in e:
            out["audit_net_kept"] = int(kept.a_net.sum())
        out.update({f"audit_{k}": v for k, v in monitor_audit(e).items()})
    if not run.trajectories.empty:
        out.update({f"val_{k}": v for k, v in forgetting(run.trajectories).items()})
    if not run.stats.empty:
        last = run.stats[run.stats["round"] == run.stats["round"].max()]
        for h in ("helpful", "harmful", "uncertain", "unused"):
            out[f"skills_{h}"] = int((last.health == h).sum())
    return out


def list_runs(runs_dir: str | Path) -> list[Path]:
    d = Path(runs_dir)
    return sorted(p for p in d.iterdir() if (p / "rounds.jsonl").exists()) if d.exists() else []
