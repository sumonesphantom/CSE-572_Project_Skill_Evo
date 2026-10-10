"""Compare runs: python analyze.py [runs/...]  -> prints a table and writes runs/summary.csv"""
import sys
from pathlib import Path

import pandas as pd

from skillevo.analysis import Run, list_runs, summarize

KEY = ["arm", "bird_test_start", "bird_test_final", "bird_test_gain", "bird_test_aulc", "spider_transfer_final",
       "val_max_drawdown", "val_forgetting_rate", "edits", "edits_kept", "edits_positive", "edits_negative",
       "kept_negative", "net_drift_kept", "audit_net_kept", "audit_sign_agreement", "audit_kappa",
       "rollbacks", "library_final", "skills_helpful", "skills_harmful"]

paths = [Path(p) for p in sys.argv[1:]] or list_runs("runs")
df = pd.DataFrame([summarize(Run(p)) for p in paths]).set_index("run")
pd.set_option("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.3f}".format)
print(df[[c for c in KEY if c in df]].T)
Path("runs").mkdir(exist_ok=True)
df.to_csv("runs/summary.csv")
print("\nfull metrics -> runs/summary.csv")
