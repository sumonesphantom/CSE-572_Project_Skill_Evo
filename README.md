# SkillEvo — self-evolving agent skills with an evolution-quality monitor

CSE 572 (Fall 2026) final project. A Text-to-SQL agent keeps a library of natural-language
**skills**. After every batch of episodes an evolver LLM proposes lifecycle operations
(**create / update / merge / discard**). A **monitor** decides, with paired statistics, whether
each change moves the agent in a **positive or negative** direction, and records why.

```
 train episodes ──► Evolver (glm-4.7-flash) ──► proposed ops ──► Monitor ──► keep / reject
      ▲                                                            │  per-edit McNemar A/B on val probe
      │                                                            │  per-skill lift + leave-one-out
  Solver (qwen2.5:7b) + retrieved skills ◄── Skill library ◄───────┘  round drawdown → rollback
                                                 │ every version + diff + verdict logged
                                                 ▼
                                      Streamlit dashboard (runs/*)
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct && ollama pull glm-4.7-flash
# data (~5 GB unzipped, git-ignored)
curl -L -o minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip && unzip minidev.zip -d data/raw/bird
.venv/bin/gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O spider.zip && unzip spider.zip -d data/raw/spider
.venv/bin/python scripts/prepare_data.py       # -> data/processed/{bird,spider}.jsonl (splits.json is committed)
```

## Run

```bash
.venv/bin/python -m pytest -q tests                                  # offline, fake LLMs, ~1 s
.venv/bin/python run.py --arm guarded --config config.fast.yaml --name quick   # ~30-60 min smoke run
.venv/bin/python run.py --arm all                                    # the 4 arms, full config
.venv/bin/python run.py --arm guarded --seed 1                       # repeat with other seeds
.venv/bin/python analyze.py                                          # comparison table -> runs/summary.csv
.venv/bin/streamlit run dashboard/app.py                             # dashboard
```

All LLM calls are temperature 0 and cached in `data/cache/llm_cache.sqlite`, so a crashed run
re-run with the same name/seed fast-forwards through everything already computed, and identical
(task, skill-context) pairs are never solved twice across arms.

## Experiment arms

| arm | operations | edit acceptance | evolver sees stats | rollback |
|---|---|---|---|---|
| `noskill` | – | – | – | – |
| `append_only` | create | all kept | no | no |
| `unguarded` | create/update/merge/discard | all kept (monitor only observes) | no | no |
| `gate_only` | create/update/merge/discard | kept only if full-val accuracy strictly improves (SkillOpt-style) | no | no |
| `guarded` | create/update/merge/discard (+ monitor discards) | monitor verdict | yes | yes |

## How "positive vs negative" is decided

* **Edit level** (`monitor.evaluate_edit`): old vs new library on a probe of validation tasks
  (tasks that retrieve the touched skills + random tasks for collateral damage). Outcome flips
  → `gained`, `lost`, `net`, exact McNemar *p*. Label: `positive` (net ≥ 2), `negative` (net ≤ −2),
  `neutral`, or `inert` (never retrieved).
* **Audit**: the same edit is re-tested on held-out *test* tasks (`monitor.audit_split`); agreement
  (sign agreement, Cohen's κ, precision of "positive") measures whether the monitor is trustworthy.
* **Skill level**: lift = success with library − cached no-skill success on the tasks where the
  skill was retrieved (bootstrap CI), plus leave-one-out marginal effect. Health:
  helpful / harmful / uncertain / unused. (Lift alone is confounded by co-retrieved skills;
  leave-one-out is what catches a harmful skill riding on a helpful one — see tests.)
* **Round level**: val accuracy vs previous round and vs best snapshot (paired); max drawdown and
  forgetting rate summarize negative drift; the guarded arm rolls back on net ≤ −3 vs best.

## Logs (`runs/<name>/`)

`rounds.jsonl` accuracy/size/verdict counts per round · `edits.jsonl` every proposed edit with
verdict, audit and kept flag · `skill_versions.jsonl` every version with parents and diff ·
`skill_stats.jsonl` per-skill lift/LOO/health per round · `trajectories.jsonl` every episode ·
`evolver.jsonl` raw evolver outputs · `library/round_XX.json` library snapshots.
