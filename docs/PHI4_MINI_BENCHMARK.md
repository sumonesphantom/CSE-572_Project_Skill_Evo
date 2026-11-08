# Phi-4-mini benchmark: BIRD mini-dev and Spider (CSE 572)

Owner: Srikar. Branch `srikar/phi4-mini-benchmark` (from `main`). Model: **Phi-4-mini-instruct (3.8B)**,
Ollama `phi4-mini` (Q4_K_M). Everything is free and local: Ollama and llama.cpp on an 8 GB MacBook Air, plus a
free Kaggle T4 GPU for fine-tuning. No paid APIs.

## Summary

| Setup (test sets opened once, pre-registered) | BIRD test (155) | Spider test (200) |
|---|---|---|
| Baseline: repo prompt | 25.8 | 59.0 |
| **E1: schema-aware repair + 3 examples (`b1_fs3`)** | **29.7** [22.6–36.8] | 65.5 [58.5–72.0] |
| **E2: LoRA fine-tune (D1) + `b1_fs3`** | 23.2 [16.8–29.7] | **72.0** [65.5–78.5] |
| Venkat, Qwen2.5-Coder-1.5B best | 29.0 | 66.0 |
| Venkat, Qwen2.5-Coder-3B best | 38.7 | 76.5 |
| Venkat, Qwen2.5-Coder-7B best | 58.7 | 83.5 |

EX %, 95% bootstrap interval in brackets. Venkat's numbers come from `origin/venkat/qwen-benchmark:results/bench_results.csv` (same splits, same EX metric).

1. **Schema-aware repair is the one robust gain.** When a query fails, the harness looks the problem up in the
   database and tells the model what is wrong ("T1 is `satscores`, which has no `School`; `School` is in
   `schools`, your T2"). Result: +6 questions and 0 lost on BIRD val, then the same +6 and 0 lost on BIRD test
   (p = 0.031 on both).
2. **Fine-tuning helps Spider and hurts BIRD.** The training data is mostly Spider. Spider test gains +26 over the
   repo prompt (p = 0.002). On BIRD, the +7 seen on val did not hold: on test the fine-tuned model is 10 questions
   below E1 (p = 0.076).
3. **Against Qwen:** Phi-4-mini matches Coder-1.5B on BIRD and beats it on Spider. It stays below Coder-3B by
   9 points on BIRD and 4.5 on Spider, so the primary target was **not** reached. It is a general model with no
   SQL-specific pretraining.
4. **Nothing that adds prompt text helped**: value retrieval, a plan before the SQL, more examples, examples chosen
   by embeddings. **SynSQL data hurt BIRD.** **Self-evolved skills (SkillEvo):** 0 kept in 4 arms, including two
   with our fixes.

## 1. Setup

- **Data:** the repo's own splits (`data/processed/splits.json`): `bird_val` 98, `bird_test` 155, `spider_test`
  = the seeded 200 of `spider:test` used in `config.yaml`. Decisions use `bird_val` only. The test sets are sealed:
  the runner refuses them without `--allow-test`. They were opened for the Phase A baseline rungs (fixed in
  advance) and for the two pre-registered Phase E setups (`results/phaseE_preregistration.md`).
- **Harness:** `phibench/`, written for this branch. It handles the prompt strategies, a resumable runner, a
  response cache keyed on every setting that can change an answer, and the metrics: EX with a bootstrap CI,
  Soft-F1, R-VES, error categories, and paired McNemar tests. 48 offline tests.
- **Statistics:** every claim compares two runs on the same questions (gained / lost / exact McNemar p). On 98–200
  questions, a few points is noise.

## 2. Phase A: baseline ladder

| Rung | BIRD val | BIRD test | Spider test |
|---|---|---|---|
| a1 repo prompt (DDL + 2 rows, 6k cap) | 27.6 | 25.8 | 59.0 |
| a2 compact schema (types, PKs, examples, FKs) | 25.5 | 23.2 | 57.0 |
| a3 + output guidelines | 28.6 | 23.2 | 60.5 |
| a4 + 3 retrieved examples | 28.6 | 25.8 | 65.0 |

No neighbouring step is significant on BIRD. Examples help Spider (a1 → a4: +12, p = 0.065) but not BIRD
(17 gained / 17 lost on test), unlike Qwen-Coder-7B where they were the biggest lever.

**Error analysis (a4, BIRD test):** 44 of 155 answers fail with a column error. In 37 of those the column exists
but in another table: the model puts the right column on the wrong alias. The plain repair turn fixed 2 of 78.

## 3. Phase B: prompt improvements (bird_val, paired against the parent setup)

| Step | Change | Gained / lost | p | Kept |
|---|---|---|---|---|
| **B1** | schema-aware repair (2 rounds + retry on empty result), on a4 | **+6 / −0** | **0.031** | **yes → `b1_fs3`, 34.7** |
| B1 | same on a3 (`b1_rules`) | +4 / −0 | 0.125 | – |
| B2 | stored values matching the question, in prompt + retry | +8 / −10 | 0.82 | no |
| B2 | longest match per phrase only | +7 / −10 | 0.63 | no |
| B2 | values in the empty-result retry only | 0 / 0 | 1.0 | no |
| B4 | short plan (table.column, joins, filters) before the SQL | +12 / −18 | 0.36 | no (4× slower) |
| B3 | examples rewritten from T1/T2 to real table names | +13 / −11 | 0.84 | no (rule: p < 0.05) |
| B3 | 5 examples | +8 / −12 | 0.50 | no |
| B3 | examples chosen by local embeddings (nomic-embed-text) | +9 / −11 | 0.82 | no |

Value retrieval did fix spellings ('Fresno Unified', 'Durability', '201306'): empty results fell from 11 to 4.
It broke about as many answers through near-miss values. The plan step was not followed (Phi wrote generic
numbered steps). This matches Venkat's finding that reasoning before SQL hurts small Qwen models.

## 4. Phase D: LoRA fine-tuning

An 8 GB Mac cannot train past ~2k tokens (Phi-4-mini's 200k vocabulary makes the logits huge), which would drop
most BIRD schemas. Training therefore ran on a free Kaggle T4 (`finetune/kaggle_train.py`, notebook
`finetune/phi4mini_kaggle.ipynb`):

- **Training setup:** LoRA rank 16 (alpha 32) on the last 16 layers, fp16 base, 400 steps at effective batch 8,
  lr 1e-4 cosine. The loss covers the SQL answer only, with the output layer evaluated on answer positions only,
  so 8k-token prompts fit. Each training prompt is the eval prompt (a3).
- **Data:**
  - `d1`: Spider train (3,000 sampled) + `spider:train` + `bird:train` ×3, 4,046 examples.
  - `d2`: `d1` + 4,000 SynSQL-2.5M examples (Apache-2.0, synthetic databases), 8,046 examples.
  - Evaluation questions and over-long examples were removed.
- **Training run:** about 610 tok/s, 111 / 130 min. `spider:val` loss 2.62 → 0.201 (d1) / 0.215 (d2).
- **Conversion:** the merged model was converted with llama.cpp to the **same Q4_K_M as Ollama's phi4-mini**. The
  untrained base went through the identical pipeline as the control.
- **Numerics check:** fp16 matches an fp32 CPU reference to 0.14% (`--diagnose`).

| bird_val | b1_rules | b1_fs3 |
|---|---|---|
| official phi4-mini | 32.7 | 34.7 |
| base (our conversion) | 32.7 (7/7 vs official) | 30.6 (9/13) |
| d1 | 31.6 (−1 vs base) | **37.8** (+7, p = 0.21) |
| d2 | 22.4 (−10, p = 0.064) | 33.7 (+3) |

SynSQL (d2) hurt BIRD. Its synthetic style and DDL schemas appear to pull the model away from BIRD's
annotation habits.

## 5. Phase C: SkillEvo with Phi-4-mini as solver and evolver

Budget as in Venkat's Qwen runs (4 rounds × 30 episodes, 20 + 10 probe questions per edit). Two fixes for why
his run kept 0 of 16 skills, both off by default in `skillevo/`:

- **C1 rejection memory:** the evolver sees its rejected proposals with their verdicts.
- **C2 database skills:** a skill tied to one database is always offered to that database's questions.

| Arm | Skills tested | Relevant probe questions per skill | Fixed / broke | Kept |
|---|---|---|---|---|
| guarded (repo) | 6 | 2–6 | 2 / 2 | 0 |
| guarded + C1/C2 in the prompt | 8 | 0–10 (median 1.5) | 1 / 2 | 0 |
| guarded + C1/C2 enforced by the harness | 4 | 3–11 (median 6.5) | 1 / 2 | 0 |

All arms: BIRD val 27.6, BIRD test 25.2, Spider 59.0 (the library stays empty).

With the fixes in the prompt only, Phi never used the `db` field, re-proposed rejected skills 4 times, and wrote
non-SQLite functions (`CONVERT_DATE`). With the harness enforcing the fixes, it:

- dropped repeats and non-SQLite skills;
- tagged skills to a database by the tables their SQL names.

Each skill then got about 3× more relevant probe questions. The skills themselves (generic percentage and
difference recipes) still did not help. With a small evolver, **skill quality is the bottleneck**, not delivery.
The monitor did its job: nothing harmful was kept and no score dropped. Logs: `results/phaseC/`.

## 6. Phase E: final test run

| Comparison (paired) | BIRD test | Spider test |
|---|---|---|
| E1 vs a4 (Phase A best) | **+6 / −0, p = 0.031** | +5 / −4 |
| E2 vs E1 | +8 / −18 (−10, p = 0.076) | +33 / −20 (+13, p = 0.098) |
| E2 vs a1 repo prompt | +14 / −18 | **+46 / −20 (+26, p = 0.002)** |

By difficulty on BIRD test (simple / moderate / challenging):

- E1: 50.0 / 24.4 / 15.2
- E2: 40.9 / 15.4 / 18.2

On Spider simple, E2 reaches 87.7.

## 7. Data issues found

- BIRD mini-dev lists questions 137 and 138 twice. The repo split puts one copy of `bird-138` in train and one in
  val, and counts `bird-137` twice in test (n = 155). We keep the split for comparability, but remove
  eval-overlapping questions from every few-shot pool and training set. In Venkat's few-shot val runs,
  `bird-138` could retrieve its own copy.
- `prepare_data.py` dropped `bird-701` once on the Mac (a one-off gold-query timeout). Rerunning it reproduced the
  committed splits exactly.

## 8. Caveats

- Small evaluation sets (±7–9 points CI). Only the paired tests above should be read as evidence.
- Several B variants were tried on the same 98 val questions. We kept a strict rule (p < 0.05) and stopped tuning
  after each round, but some selection bias remains. E2's val gain not holding on test is an example.
- Few-shot pools and BIRD training data come from the same 11 databases as BIRD test (the repo's split design; the
  official BIRD setting uses unseen databases).
- The fine-tuned model was trained in fp16 on top of the bf16-trained base and served as Q4_K_M.
  Our base conversion matches Ollama's phi4-mini within noise.
- EX uses set semantics, as in the repo.

## 9. Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # + data as in README.md
ollama pull phi4-mini && ollama pull nomic-embed-text
.venv/bin/python -m pytest -q tests                                  # 48 offline tests
.venv/bin/python -m phibench.run --model phi4-mini --strategies a1_repo,a2_mschema,a3_rules,a4_fs3 --sets bird_val
.venv/bin/python -m phibench.run --model phi4-mini --strategies b1_fs3 --sets bird_val
bash scripts/phi_lora.sh package    # then Kaggle: finetune/phi4mini_kaggle.ipynb; import the GGUFs:
bash scripts/phi_lora.sh import d1 phi4mini-d1-Q4_K_M.gguf && bash scripts/phi_lora.sh evalo d1
caffeinate -i bash scripts/phi_skillevo.sh                           # Phase C arms
.venv/bin/python -m phibench.compare --venkat                        # tables + head-to-head
```
