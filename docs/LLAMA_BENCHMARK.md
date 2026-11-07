# Llama 3.2 1B (Ollama `llama3.2:1b`): Text-to-SQL benchmark

**Model:** Llama 3.2 1B Instruct, served by Ollama (Q8_0), on a MacBook Air M5 with 16 GB RAM.
**Data:** BIRD mini-dev and Spider dev, using the team repo's own splits (`data/processed/splits.json`, unchanged).
**Metric:** execution accuracy (EX), using the repo's `skillevo.sqlexec.is_correct`. A query counts as correct if its result set equals the gold query's result set (order ignored).

## Headline

| | BIRD test (155) | Spider test (200) |
|---|---|---|
| Team baseline (repo pipeline, no skills) | 9.0% | 42.0% |
| **Best result (base model + pruned schema + 5-vote)** | **11.0%** | **50.0%** (p = 0.014) |
| Fine-tuned (LoRA) + 5-vote | 4.5% | 52.0% (p = 0.029) |

- **Best overall: the base model with better prompting.**
  - Spider improves by +8 points, which is statistically significant.
  - BIRD improves by +2 points, which is within noise.
- **Fine-tuning helped Spider only slightly.** On Spider test it scored 52.0% against 50.0% for the best prompt setup, which is not a significant difference.
- **Fine-tuning badly hurt BIRD**, dropping it from 11.0% to 4.5%. The fine-tuned model is not recommended.

Every setting was chosen on the **validation** sets only (BIRD val 98, Spider val 100). The test sets were run once at the end.

## Full results (EX %)

p-values are exact McNemar tests on the same questions, compared with the team baseline. p < 0.05 means the difference is unlikely to be chance.

| Setup | BIRD val | Spider val | **BIRD test** | **Spider test** | sec/q | LLM calls/q |
|---|---|---|---|---|---|---|
| Team baseline (repo, no skills) | 9.2 | – | 9.0 | 42.0 | – | – |
| repo pipeline, reimplemented (sanity check) | 9.2 | 43.0 | – | – | 6.8 | 1.5 |
| + robust SQL extraction | 9.2 | 43.0 | – | – | 7.2 | 1.5 |
| + compact full schema (no 6,000-char cut) | 12.2 | 44.0 | – | – | 6.4 | 1.6 |
| + schema linking (pruned, top-4 tables + BIRD column descriptions) | 16.3 | 47.0 | 8.4 (p=1.000) | 45.5 (p=0.349) | 6.9 | 1.5 |
| + 3 few-shot examples | 19.4 | 38.0 | – | – | 7.3 | 1.4 |
| + 1 few-shot example | 14.3 | 44.0 | – | – | 6.9 | 1.3 |
| + few-shot + 5-vote | 20.4 | 39.0 | – | – | 28.1 | 5.2 |
| **pruned + 5-vote (sc5_nofs): chosen** | **21.4** | **53.0** | **11.0** (p=0.607) | **50.0** (p=0.014) | 21.4 | 5.3 |
| Fine-tuned (LoRA), pruned prompt | 1.0 | 42.0 | 3.9 (p=0.096) | 42.0 (p=1.000) | 6.0 | 1.6 |
| Fine-tuned (LoRA), pruned + 5-vote | 4.1 | 46.0 | 4.5 (p=0.167) | 52.0 (p=0.029) | 21.4 | 5.4 |

"5-vote" means the model answers five times: one greedy answer and four sampled at temperature 0.8. The queries that run are grouped by their results, and the most common result wins.

### Extra metrics (test sets)

"Simple / Moderate / Challenging" are the difficulty labels in the repo's processed data, for both datasets.

| Setup | Set | EX % | Valid SQL % | Simple | Moderate | Challenging | Most common errors |
|---|---|---|---|---|---|---|---|
| pruned | BIRD test | 8.4 | 35 | 22.7 | 2.6 | 3.0 | wrong column 78, wrong result 41, syntax 10 |
| pruned | Spider test | 45.5 | 74 | 57.0 | 32.8 | 22.7 | wrong result 56, wrong column 48 |
| **sc5_nofs (chosen)** | BIRD test | 11.0 | 55 | 25.0 | 6.4 | 3.0 | wrong result 69, wrong column 54 |
| **sc5_nofs (chosen)** | Spider test | 50.0 | 85 | 63.2 | 35.9 | 22.7 | wrong result 70, wrong column 28 |
| Fine-tuned, pruned | BIRD test | 3.9 | 28 | 11.4 | 1.3 | 0.0 | wrong column 83, wrong result 38, syntax 23 |
| Fine-tuned, pruned | Spider test | 42.0 | 66 | 50.9 | 37.5 | 9.1 | wrong column 49, wrong result 47 |
| Fine-tuned, sc5_nofs | BIRD test | 4.5 | 45 | 13.6 | 1.3 | 0.0 | wrong column 62, wrong result 62, syntax 20 |
| Fine-tuned, sc5_nofs | Spider test | 52.0 | 84 | 61.4 | 46.9 | 18.2 | wrong result 63, wrong column 22 |

- **Valid SQL %** is the share of answers that run without error. 5-vote raises it a lot: from 35% to 55% on BIRD and from 74% to 85% on Spider.
- **Cost:** 5-vote takes about 3× the time (about 21 s per question against 7 s).

## What each step does

1. **Robust extraction.** Pulls the SQL out of the model's reply even when the code fence is broken, and keeps only the first statement. It made no difference for the 1B model, but it removes a failure mode.
2. **Compact full schema.** The repo cuts the schema text at 6,000 characters, which cut off 77 of the 155 BIRD test questions. The compact format fits every schema without truncation.
3. **Schema linking.** Keeps only the 4 tables most related to the question, plus the tables their foreign keys point to, and adds BIRD's column descriptions. The right tables are kept 96% of the time (measured on BIRD train).
4. **Few-shot examples.** Similar solved questions from the training split. They helped BIRD val but hurt Spider val, so they were dropped.
5. **5-vote (self-consistency).** Described above. It was the most reliable gain on both validation sets.

## Fine-tuning experiment (LoRA)

- **Data:** 8,702 examples.
  - Spider `train_spider` + `train_others` (their databases do not overlap with Spider dev).
  - BIRD mini-dev train split: 247 questions.
  - Prompt format identical to sc5_nofs; the target is the gold SQL.
- **Training:** MLX-LM LoRA.
  - Last 16 layers, 5.6M trainable parameters (0.46%).
  - 1,000 iterations, batch size 4, learning rate 1e-4, prompt masked, max length 1,024.
  - About 50 minutes on the M5.
- **Validation loss:** 1.70 → 0.32 (step 200) → 0.19 (step 600, the best checkpoint, which was the one used).
- **Deployment:** merged into the base model and imported into Ollama as `llama3.2-1b-sql` with int8 quantization. Ollama 0.40 no longer offers `q8_0`, so the base model is Q8_0 while the fine-tuned one is int8.

**Why it failed on BIRD.** I tested the model directly in MLX, without Ollama, on the first 40 BIRD val questions:

| Model (MLX, greedy, pruned prompt) | First 40 BIRD val |
|---|---|
| Base Llama 3.2 1B | 5 / 40 (12.5%) |
| LoRA, step 200 | 0 / 40 |
| LoRA, step 600 (merged) | 0 / 40 |

- **The Ollama import is not the cause:** the model is just as bad in MLX.
- **The training data is the cause.**
  - 97% of the examples were Spider: simple queries, no hints, and `T1`/`T2` table aliases.
  - The model learned that style and applies it to BIRD. It invents short column names, for example `T1.cds` where the real column is `CDSCode`, and it ignores the hint. "Wrong column" errors rise sharply.
  - The damage is already there at step 200, so training for fewer steps does not fix it.
- **On Spider it helped a little** (52.0% against 50.0%, not significant), because there it matches the style.

**What would likely fix it (not run, time limit):**
- Train on BIRD's official train set (about 9.4k questions with hints), or weight BIRD examples much more heavily.
- Use a lower learning rate (about 2e-5).

## Notes on the team repo setup (found while benchmarking)

1. **Schema truncation:** `schema_text(max_chars=6000)` cuts large BIRD schemas, affecting 77 of the 155 BIRD test questions.
2. **Duplicate questions:** BIRD mini-dev questions 137 and 138 are duplicates and land in more than one split. They were removed from the few-shot and training pools, so no test question leaks into training.
3. **`card_games.sqlite` journal mode:** the database ships in WAL mode, which cannot be opened read-only, so 52 questions were silently dropped. Fixed with `PRAGMA journal_mode=delete`.
4. **Quantization:** the base model is Q8_0 and the fine-tuned model is int8 (see above).

## Step labels in `bench_results.csv`

| rung | strategy | What it adds |
|---|---|---|
| L0 | repo | Repo prompt + repair turn |
| L1 | extract | Robust SQL extraction |
| L2 | compact | Full compact schema, no 6,000-character cut |
| L3 | pruned | Schema linking + BIRD column descriptions |
| L4 / L4b | fewshot / fewshot1 | 3 / 1 BM25 examples |
| L5 | sc5 | Few-shot + 5-sample vote |
| L5b | sc5_nofs | Pruned + 5-sample vote (**chosen**) |
| +FT | – | Same prompt on the LoRA fine-tuned model |

## Reproduce

Everything lives in `bench_llama/`; results in `results/llama1b/`.

```bash
python scripts/prepare_data.py                       # datasets (see main README)
sqlite3 data/raw/bird/minidev/MINIDEV/dev_databases/card_games/card_games.sqlite "PRAGMA journal_mode=delete;"
ollama pull llama3.2:1b
python -m bench_llama.evaluate --variants sc5_nofs --sets val,spider_val,bird_test,spider_transfer --model llama3.2:1b
python bench_llama/report.py                         # rebuilds the tables above
python -m bench_llama.export_csv                     # results/llama1b/bench_results.csv (team CSV columns)
```

- **Per-question outputs:** `results/llama1b/runs_bench/<model>__<variant>/<set>.jsonl`, with summaries in `<set>.summary.json`.
- **Team baseline (repo `run.py`, no skills):** `results/llama1b/baseline_noskill/`; config in `configs/llama/llama3.2-1b.yaml`.
- **Comparison CSV:** `results/llama1b/bench_results.csv`, same columns as the team's `bench_results.csv`. Extra metrics: bootstrap 95% CI, first-attempt EX, oracle EX (any of the 5 candidates correct), BIRD soft-F1. Token counts and R-VES are not measured (empty).
- **Fine-tuning** (needs a separate Python 3.12 venv with `mlx-lm`, Apple Silicon):
  - `bench_llama/make_train_data.py` builds the training data.
  - `bench_llama/train_after.sh` runs the LoRA training.
  - `bench_llama/import_finetuned.sh` merges the model and imports it into Ollama.
  - `bench_llama/mlx_check.py` is the direct MLX check.
