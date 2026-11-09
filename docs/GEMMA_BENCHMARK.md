# Gemma 4 E2B benchmark: BIRD mini-dev and Spider

Owner: Manan. Branch `manan/gemma-benchmark` (from `kamalesh/smollm2-benchmark` at `787ecda`, which holds the
team harness `bench/`). Model: **Gemma 4 E2B Instruct (4.6B)**, Ollama `gemma4:e2b-it-q4_K_M` (Q4_K_M),
Ollama 0.40.15 on an RTX 3070 Laptop GPU (8 GB), Windows.

## Summary

| Setup | BIRD val (98) | BIRD test (155) | Spider test (200) |
|---|---|---|---|
| P1 repo prompt, repo SQL extractor | 19.4 | 18.1 | 33.5 |
| **S1 repo prompt, robust SQL extractor** | 40.8 | **42.6** [34.8–50.3] | **74.0** [68.0–80.0] |
| S6 pruned schema + 3 retrieved examples | 46.9 | 45.2 [36.8–52.9] | 65.5 [59.0–71.5] |

EX %, 95% bootstrap interval in brackets. Same splits (`data/processed/splits.json`), same EX scorer, one repair
turn and temperature 0, as in the other branches.

1. **The repo extractor breaks on Gemma's output.** Gemma answers in a ` ```sqlite ` code fence. The repo regex
   `(?:sql|sqlite)?` matches `sql` first and leaves `ite` in front of the query, so SQLite rejects it:
   97 of 155 BIRD test answers and 114 of 200 Spider test answers start with `ite`. P1 measures this bug, not the
   model. S1 is the same prompt and the same cached model answers with `bench/extract.py` (the extractor from the
   Llama 1B benchmark): +38 / −0 on BIRD test, +81 / −0 on Spider test.
2. **Gemma 4 E2B is the strongest sub-5B solver on the repo prompt.** With S1 it scores 42.6 / 74.0 on BIRD test /
   Spider test, against 35.5 / 69.0 for Qwen2.5-Coder-3B, 25.8 / 59.0 for Phi-4-mini and 41.9 / 72.0 for
   Qwen2.5-7B-Instruct (other branches' numbers, repo prompt, repo extractor).
3. **Retrieved examples do not help reliably.** S6 against S1, paired on the same questions: BIRD val +15 / −9
   (p = 0.31), BIRD test +21 / −17 (p = 0.63), Spider test +13 / −30 (**−17, p = 0.014**). The BIRD gain is within
   noise and Spider gets worse. S6 changes two things at once (schema pruning and examples), so this run does not
   say which one costs Spider.

## By difficulty and error type (test sets)

| Setup | Set | Valid SQL % | Simple | Moderate | Challenging | Errors |
|---|---|---|---|---|---|---|
| S1 | BIRD test | 88 | 61.4 | 38.5 | 27.3 | wrong result 71, wrong column 13, syntax 3 |
| S1 | Spider test | 97 | 79.8 | 70.3 | 54.5 | wrong result 46, wrong column 4 |
| S6 | BIRD test | 97 | 59.1 | 41.0 | 36.4 | wrong result 80, wrong column 3 |
| S6 | Spider test | 99 | 69.3 | 65.6 | 45.5 | wrong result 67, wrong column 1 |

Most remaining errors are queries that run and return the wrong rows, the same pattern as Qwen2.5-Coder-7B.

## What was and was not run

- Two configurations were fixed in advance (repo prompt, and pruned schema + 3 examples) and run once on the
  validation and test sets. Nothing was selected on the test sets; the full validation ladder
  (`scripts/gemma_pipeline.py`) was not run.
- R-VES was not timed (`--no-ves`). No fine-tuning and no SkillEvo arms on this branch.
- One run per configuration (temperature 0, cached). On 98–200 questions the intervals are about ±7–10 points.
- Earlier Colab pilots (`01_gemma_prompting.ipynb`, `02_gemma_qlora.ipynb`) used the Hugging Face checkpoint
  `google/gemma-4-E2B-it` on a separate 1,131-question validation split and are not comparable to this table.

## Notes on the setup

- `scripts/prepare_data.py` dropped `bird-701` on this machine (its gold query exceeded the 120 s limit), which
  rewrites the splits. The committed `splits.json` was restored and all 500 BIRD questions kept, so the eval sets
  are the team's. `bird-701` is in `bird_test`.
- `card_games.sqlite` ships in WAL mode; it was switched with `PRAGMA journal_mode=delete` as in the Llama branch.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt     # + data as in README.md
ollama pull gemma4:e2b-it-q4_K_M
python -m bench.run --model gemma4:e2b-it-q4_K_M --strategy zs_repo --sets bird_val,bird_test,spider_test --no-ves
python -m bench.run --model gemma4:e2b-it-q4_K_M --strategy sm_extract --sets bird_val,bird_test,spider_test --no-ves
python -m bench.run --model gemma4:e2b-it-q4_K_M --strategy sm_pruned_fs3 --sets bird_val,bird_test,spider_test --no-ves
python -m bench.smol_report --prefix gemma4 --out results/gemma4_e2b   # CSV, tables, per-question outputs
```

Results: `results/gemma4_e2b/bench_results.csv` (team CSV columns), `tables.md`, and per-question outputs under
`runs_bench/`. `notebooks/gemma_benchmark_colab.ipynb` runs the same harness on a Colab T4.
