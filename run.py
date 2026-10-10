"""Run a self-evolution experiment.

    python run.py --arm guarded                  # monitor-gated evolution
    python run.py --arm unguarded --rounds 8     # same evolver, no gate (observe drift)
    python run.py --arm all                      # noskill, append_only, unguarded, guarded
    python run.py --arm guarded --config config.fast.yaml --name quick-test
"""
import argparse
import sys
from pathlib import Path

import yaml

from skillevo.runner import ARMS, Experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="guarded", choices=[*ARMS, "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--name", help="run directory name (default: <arm>-s<seed>)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["root"] = str(Path(__file__).resolve().parent)
    if a.seed is not None:
        cfg["seed"] = a.seed
    arms = ARMS if a.arm == "all" else [a.arm]
    for arm in arms:
        name = a.name if a.name and len(arms) == 1 else (f"{a.name}-{arm}" if a.name else None)
        Experiment(cfg, arm, name, log=lambda m: print(m, flush=True)).run(a.rounds)


if __name__ == "__main__":
    sys.exit(main())
