"""The Qwen benchmark plan: one place that lists every experiment, its command and its output dir.
The Excel tracker's Experiment_Plan sheet is generated from this list, and `status` is read from disk.

    python -m bench.plan list                    # table of experiments + status
    python -m bench.plan run A                   # run every pending experiment of phase A, in order
    python -m bench.plan run A03,B01             # run specific experiments
    python -m bench.plan run A --dry             # print the commands only

Phases
  A  prompt ladder on Qwen2.5-Coder-7B (selection on bird_val, report bird_test / spider_test)
  B  model sweep: same key rungs on Coder-3B, Coder-1.5B, Qwen2.5-7B-Instruct
  C  self-evolving skills (SkillEvo arms), Qwen evolver vs GLM evolver
  D  LoRA fine-tuning (MLX) + MLX base-model controls
  E  combined: best model/adapter + best prompt + best evolved library
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SETS = "bird_val,bird_test,spider_test"
RUNGS = ["zs_ddl", "zs_repo", "zs_mschema", "zs_desc", "zs_rules", "fs3", "fs3_cot", "fs3_sc5", "fs3_cot_sc5"]
MODELS = {"coder7b": "qwen2.5-coder:7b", "coder3b": "qwen2.5-coder:3b", "coder1.5b": "qwen2.5-coder:1.5b",
          "instruct7b": "qwen2.5:7b-instruct"}
MLX = {"coder7b": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", "coder3b": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
       "coder1.5b": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"}


@dataclass
class Exp:
    id: str
    family: str
    model: str
    method: str
    question: str               # what this experiment answers
    cmds: list[str]
    out: str                    # output dir, relative to ROOT (status = done if it holds results)
    est_hours: float
    needs: list[str] = field(default_factory=list)
    note: str = ""

    def status(self) -> str:
        p = ROOT / self.out
        if (p / "metrics.json").exists():
            import json
            m = json.loads((p / "metrics.json").read_text())["metrics"]
            if all(m.get(s) for s in SETS.split(",")):
                return "done"
        if (p / "rounds.jsonl").exists() and _finished(p):
            return "done"
        if p.exists() and any(p.iterdir()):
            return "partial"
        return "planned"


def _finished(run_dir: Path) -> bool:
    import json
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    rounds = [json.loads(line)["round"] for line in open(run_dir / "rounds.jsonl") if line.strip()]
    target = 0 if cfg.get("arm") == "noskill" else cfg.get("rounds", 0)  # the runner forces 0 rounds for noskill
    return bool(rounds) and max(rounds) >= target


def bench_cmd(model, rung, sets=SETS, extra=""):
    return f"{PY} -m bench.run --model {model} --rungs {rung} --sets {sets} {extra}".strip()


def bench_out(model, rung, library=None, tag=None):
    from bench.run import run_name
    return f"runs/bench/{run_name(model, rung, library, tag)}"


def plan() -> list[Exp]:
    E: list[Exp] = []
    m7 = MODELS["coder7b"]
    # ---- A: prompt ladder on the main model
    for i, r in enumerate(RUNGS):
        sc = "sc5" in r
        E.append(Exp(f"A{i:02d}", "prompt", m7, r, f"P{i}: effect of '{r}' over the previous rung",
                     [bench_cmd(m7, r)], bench_out(m7, r), 3.0 if "cot_sc5" in r else 1.6 if sc else 0.5 if "cot" in r else 0.35))
    E.append(Exp("A09", "prompt", m7, "zs_repo_full", "P1b control: is the P1->P2 gain the schema format or the 6k-char cap?",
                 [bench_cmd(m7, "zs_repo_full")], bench_out(m7, "zs_repo_full"), 0.35))
    for eid, r, h, q in (("A10", "ms_rules", 0.35, "P4m: rules on P2's schema (no descriptions)"),
                         ("A11", "ms_fs3", 0.5, "P5m: few-shot on P2's schema"),
                         ("A12", "ms_fs3_sc5", 1.6, "P7m: self-consistency on P5m")):
        E.append(Exp(eid, "prompt", m7, r, q, [bench_cmd(m7, r)], bench_out(m7, r), h))
    # ---- B: model sweep on the rungs that matter
    for j, (key, m) in enumerate((k, v) for k, v in MODELS.items() if k != "coder7b"):
        for k, r in enumerate(["zs_repo", "zs_mschema", "ms_fs3", "fs3", "fs3_sc5"]):
            E.append(Exp(f"B{j}{k}", "prompt", m, r, f"{key}: does '{r}' transfer to a {'smaller' if 'b' in key and key != 'instruct7b' else 'general'} model?",
                         [bench_cmd(m, r)], bench_out(m, r), {"coder3b": 0.2, "coder1.5b": 0.15}.get(key, 0.35) * (5 if "sc5" in r else 1.5 if "cot" in r else 1)))
    # ---- C: self-evolving skills
    arms = [("C01", "coder7b__evo-qwen7b", "guarded", "Qwen-7B evolves its own skills, monitor-gated (main arm)"),
            ("C02", "coder7b__evo-qwen7b", "unguarded", "same evolver, no gate: how much negative drift?"),
            ("C03", "coder7b__evo-qwen7b", "append_only", "accumulate-only baseline (Voyager/ExpeL style)"),
            ("C04", "coder7b__evo-qwen7b", "gate_only", "aggregate val gate (SkillOpt style) vs per-edit monitor"),
            ("C05", "coder7b__evo-glm", "guarded", "reference: bigger GLM-4.7-flash evolver, same solver"),
            ("C06", "coder3b__evo-qwen7b", "guarded", "3B solver taught by the 7B Qwen evolver"),
            ("C07", "coder3b__evo-self", "guarded", "3B solver evolving its own skills"),
            ("C08", "coder1.5b__evo-qwen7b", "guarded", "1.5B solver taught by the 7B Qwen evolver"),
            ("C09", "instruct7b__evo-self", "guarded", "general 7B instruct self-evolution (continuity with smoke runs)")]
    for eid, cfg, arm, q in arms:
        name = f"{cfg}__{arm}"
        E.append(Exp(eid, "skill-evo", cfg.split("__")[0], f"{arm} / {cfg.split('__')[1]}", q,
                     [f"{PY} run.py --arm {arm} --config configs/qwen/{cfg}.yaml --name {name}"], f"runs/{name}",
                     4.0 if "glm" in cfg else 2.7 if "7b" in cfg.split("__")[0] else 1.7))
    E.append(Exp("C10", "skill-evo", "coder7b", "noskill", "no-skill arm for the SkillEvo tables",
                 [f"{PY} run.py --arm noskill --config configs/qwen/coder7b__evo-qwen7b.yaml --name coder7b__noskill"],
                 "runs/coder7b__noskill", 0.4))
    # ---- D: LoRA fine-tuning (+ MLX base controls, same quantisation as the adapters), automated
    for k, key in enumerate(MLX):
        hrs = {"coder1.5b": 0.8, "coder3b": 1.5, "coder7b": 3.0}[key]
        E.append(Exp(f"D{k}0", "fine-tune", key, "MLX base (no adapter)", "control: MLX 4-bit base, same harness",
                     [f"bash scripts/finetune_auto.sh {key} base"], bench_out(f"mlx:{key}-base", "ms_rules"), 0.5))
        E.append(Exp(f"D{k}1", "fine-tune", key, "LoRA r16, ms_rules prompt", "SFT on Spider train 3k + BIRD train x3",
                     [f"bash scripts/finetune_auto.sh {key} lora"], bench_out(f"mlx:{key}-r16", "ms_rules"), hrs + 0.5,
                     needs=[f"D{k}0"]))
    # ---- E: combined (library = final snapshot of the main Qwen-evolved arm, C01)
    lib = "runs/coder7b__evo-qwen7b__guarded/library/round_04.json"
    E.append(Exp("E01", "combined", m7, "ms_fs3 + evolved skills (C01)", "do evolved skills add on top of the best prompt?",
                 [bench_cmd(m7, "ms_fs3", extra=f"--library {lib}")], bench_out(m7, "ms_fs3", lib), 0.5, needs=["A11", "C01"]))
    E.append(Exp("E02", "combined", m7, "zs_repo + evolved skills (C01)", "skills under the prompt they were evolved with",
                 [bench_cmd(m7, "zs_repo", extra=f"--library {lib}")], bench_out(m7, "zs_repo", lib), 0.4, needs=["C01"]))
    return E


def select(ids: str) -> list[Exp]:
    exps = plan()
    want = ids.split(",")
    return [e for e in exps if any(e.id == w or (len(w) == 1 and e.id.startswith(w)) for w in want)]


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for e in plan():
            print(f"{e.id:<4} {e.status():<8} {e.family:<10} {e.model:<20} {e.method:<34} ~{e.est_hours:.1f}h")
        return
    if sys.argv[1] == "run":
        dry = "--dry" in sys.argv
        for e in select(sys.argv[2]):
            if e.status() == "done":
                print(f"skip {e.id} (done)")
                continue
            if any(c.startswith("#") or "<" in c for c in e.cmds):
                print(f"{e.id}: manual steps (server / placeholders):\n  " + "\n  ".join(e.cmds))
                continue
            for c in e.cmds:
                print(f"[{e.id}] {c}", flush=True)
                if not dry:
                    subprocess.run(shlex.split(c), cwd=ROOT, check=True)
                if not dry and e.family == "skill-evo" and not _finished(ROOT / e.out):
                    raise SystemExit(f"{e.id} stopped before its last round")


if __name__ == "__main__":
    main()
