# Qwen benchmark: small Qwen models on BIRD mini-dev and Spider

Owner: Venkat (team assignment: small Qwen). Branch `venkat/qwen-benchmark`.
Companion tracker: `results/qwen_benchmark.xlsx` (regenerate with `python -m bench.report`).

## 1. Goal and the questions we answer

The team assignment says: get the best BIRD mini-dev and Spider results for your model with any method, and report the repo's metrics. This branch answers five questions for Qwen:

1. **Prompting.** How far does a 7B Qwen coder get on prompt engineering alone, and which single change pays the most?
2. **Model size.** Do the same prompts carry over to Qwen2.5-Coder-3B / 1.5B and to the general Qwen2.5-7B-Instruct?
3. **Self-evolution with Qwen as evolver.** Can Qwen write and curate its own skill library (SkillEvo), and how much does a larger evolver (GLM-4.7-flash) add on top?
4. **Fine-tuning.** What does a LoRA adapter (MLX, on this laptop) buy each model size?
5. **Stacking.** Do fine-tuning, the best prompt and evolved skills add up?

## 2. Choices made (and the alternatives we turned down)

| Decision | Picked | Alternatives considered | Why |
|---|---|---|---|
| Main model | Qwen2.5-Coder-7B (Ollama Q4_K_M) | Qwen3-8B, Qwen2.5-7B-Instruct | SQL-tuned, same memory and speed as the repo's 7B; Qwen3 thinking mode costs 2-4x latency |
| Size sweep | Coder-3B, Coder-1.5B | Qwen2.5-0.5B | 0.5B rarely produces executable BIRD SQL; 1.5B is the smallest useful point |
| Continuity model | Qwen2.5-7B-Instruct | none | it is the repo's current solver, so the smoke-run numbers stay comparable |
| Evolver | Qwen2.5-Coder-7B (main), GLM-4.7-flash (reference arm) | Qwen-only; Qwen + GLM critic | the ablation tells us what a 19 GB teacher adds over self-evolution |
| Fine-tuning backend | MLX-LM LoRA on 4-bit bases | PEFT on MPS; full BIRD train download | MLX trains 3-5x faster on Apple silicon; BIRD train is 9 GB and not needed for a first pass |
| Selection protocol | pick on `bird_val`, report `bird_test` + `spider_test` | report the best test number | picking on test would inflate every row |

## 3. Data, splits and metrics

Data comes from the repo's own splits (`data/processed/splits.json`, stratified by database and difficulty):

| Set | Source | n | Used for |
|---|---|---|---|
| `bird_val` | BIRD mini-dev | 98 | prompt / config selection, SkillEvo monitor probes |
| `bird_test` | BIRD mini-dev | 155 | headline BIRD number, SkillEvo audit |
| `spider_test` | Spider dev, seeded sample of the 518-question test split | 200 | transfer number (same sample as `config.yaml`) |
| `bird:train` | BIRD mini-dev | 247 | few-shot pool, SkillEvo episodes, SFT data (x3 upsampled) |
| Spider train | official `train_spider.json` | 7,000 (3,000 sampled) | few-shot pool for Spider, SFT data |

Optional sets: `bird_all` (all 500 mini-dev questions, comparable to the BIRD leaderboard for zero-shot rows only) and `spider_test_full` (518).

Metrics (`bench/metrics.py`):

- **EX** (primary, the repo's metric): `set(pred rows) == set(gold rows)`, with a bootstrap 95% CI.
- **EX first attempt**: EX before the one repair turn on a SQLite error, so the repair effect is `EX - EX_first`.
- **Soft-F1**: BIRD mini-dev's partial-credit metric (row-aligned cell overlap).
- **R-VES**: BIRD mini-dev's efficiency-weighted EX (x100). We time 5 runs and take the median; the official script uses 100, so treat R-VES differences under ~5 points as noise.
- **Executable rate, repair rate, error categories, EX by difficulty, EX per database.**
- **Cost**: prompt / output tokens and LLM seconds per question.
- **Oracle EX** for self-consistency: share of questions where any of the 5 candidates was right (the ceiling of voting).
- **Paired McNemar** between any two runs on the same questions. Treat a 2-point EX change on 98 questions as noise unless McNemar agrees.
- **SkillEvo metrics**: the repo's own (gain, AULC, drawdown, forgetting, edits kept / negative, monitor audit agreement, kappa).

## 4. Methods

### 4.1 Prompt-engineering ladder (`bench/strategies.py`)

Each rung changes one thing, so the delta between neighbours belongs to that change.

| Rung | Strategy | Adds |
|---|---|---|
| P0 | `zs_ddl` | raw `CREATE TABLE` only |
| P1 | `zs_repo` | the repo prompt: DDL + 2 sample rows (byte-identical to `skillevo/solver.py`, shares its cache) |
| P1b | `zs_repo_full` | control: P1 with the 20,000-character cap P2 uses, so P1 to P2 splits into "cap" and "format" |
| P2 | `zs_mschema` | compact typed column list, PK, 3 distinct example values per column, FK lines |
| P3 | `zs_desc` | + BIRD column and value descriptions (`database_description/*.csv`) |
| P4 | `zs_rules` | + 5 output rules (exact columns, literal evidence formulas, backticks, CAST for ratios, LIMIT only for top-k) |
| P5 | `fs3` | + 3 retrieved solved examples (BM25, same database first, train split only) |
| P6 | `fs3_cot` | + short numbered reasoning before the SQL |
| P7 | `fs3_sc5` | P5 + self-consistency: greedy + 4 samples at T=0.7, majority vote on execution results |
| P8 | `fs3_cot_sc5` | P6 + the same vote |

All rungs keep the repo's single repair turn.

**Finding before any run:** the repo prompt (`schema_text`) caps the schema at 6,000 characters. Five of the 11 BIRD databases exceed it (`california_schools`, `card_games`, `codebase_community`, `european_football_2`, `formula_1`), so P1 shows the model a cut-off schema on those. P2/P3 use a 20,000-character cap, and the bench raises `num_ctx` to 16,384 so Ollama does not truncate the prompt. P1b isolates the cap: P1 to P1b measures the extra schema, P1b to P2 measures the M-Schema format.

### 4.2 Self-evolving skills with Qwen as evolver (`configs/qwen/*.yaml`)

The SkillEvo loop is the repo's, unchanged, apart from the model choice. The Qwen profile runs 4 rounds x 30 episodes, scores the test sets at rounds 0 and 4 and uses 30-question edit probes, which brings one arm down to about 3-4 hours. The reduced plan runs C01 (Qwen evolver) and C05 (GLM evolver); the other arms are optional extras.

| Arm | Solver | Evolver | Question |
|---|---|---|---|
| C01 guarded | Coder-7B | Coder-7B | main result: monitor-gated self-evolution |
| C02 unguarded | Coder-7B | Coder-7B | how much negative drift without a gate |
| C03 append_only | Coder-7B | Coder-7B | accumulate-only baseline |
| C04 gate_only | Coder-7B | Coder-7B | aggregate gate vs per-edit monitor |
| C05 guarded | Coder-7B | GLM-4.7-flash | value of a bigger evolver |
| C06 / C08 guarded | Coder-3B / 1.5B | Coder-7B | a 7B Qwen teaching smaller Qwens |
| C07 guarded | Coder-3B | Coder-3B | small-model self-evolution |
| C09 guarded | 7B-Instruct | 7B-Instruct | continuity with the smoke runs |

Skills evolve under the repo prompt (P1), because SkillEvo's solver uses it. Experiment E01 then plugs the evolved library into the best prompt rung through `bench.run --library`.

### 4.3 LoRA fine-tuning (`finetune/`)

- Data: `finetune/build_sft.py` builds chat examples from 3,000 Spider-train questions, our 309 Spider-train-split questions and the 247 BIRD train-split questions x3, using the exact prompt the model sees at eval time: `ms_rules` (P4m: P2's schema + rules, no descriptions), chosen after descriptions hurt on `bird_val`. It drops examples longer than the LoRA `max_seq_length` (4,096), because mlx_lm would otherwise cut the end of the sequence, which is the SQL answer. With descriptions (`zs_rules`) 226 of 4,049 examples were too long; the build log reports the count for `ms_rules`. The training-loss validation slice comes from `spider:val`, which no eval set uses.
- Reduced plan: Coder-3B only (D10 control, D11 adapter), run unattended by `scripts/finetune_auto.sh`. Smoke-test it first with `SMOKE=1 bash scripts/finetune_auto.sh coder1.5b lora` (~5 min).
- Training: `mlx_lm.lora -c finetune/configs/<model>_r16.yaml` (QLoRA on the MLX 4-bit base, rank 16 on the last 16 layers, loss on the SQL answer only, effective batch 8, cosine LR from 1e-4).
- Serving and eval: `mlx_lm.server --model <base> --adapter-path finetune/adapters/<model>-r16 --port 8080`, then `python -m bench.run --model mlx:<model>-r16@localhost:8080 --rungs ms_rules,ms_fs3`.
- Control: the same MLX base without an adapter (D00/D10/D20). Ollama's Q4_K_M and MLX's 4-bit differ, so compare the adapter only with its own MLX base.
- Grid of further runs (rank, layers, LR, data mix, prompt masking, DoRA) lives in the `FineTune_Grid` sheet.

### 4.4 Combined (E01, E02)

Best prompt rung + best evolved library on Coder-7B, then the best adapter + best prompt + best library.

## 5. Results

Tables between the AUTO markers are regenerated by `python -m bench.report`. Edit text outside them.

### 5.1 All benchmark runs (EX %, 95% CI in brackets)

<!-- AUTO:RESULTS -->
| Model | Rung | Strategy | Skills | BIRD val EX | BIRD test EX (95% CI) | BIRD test Soft-F1 | BIRD test R-VES | Spider test EX (95% CI) |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder:7b | P0 | zs_ddl | – | 54.1 | 47.1 (39.4–54.8) | 52.9 | 42.7 | 75.0 (68.5–81.0) |
| qwen2.5-coder:7b | P1 | zs_repo | coder7b__evo-qwen7b__guarded | 56.1 | 46.5 (38.7–54.2) | 55.1 | 43.4 | 79.0 (73.0–84.5) |
| qwen2.5-coder:7b | P1 | zs_repo | – | 56.1 | 46.5 (38.7–54.2) | 55.1 | 42.2 | 79.0 (73.0–84.5) |
| qwen2.5-coder:7b | P1b | zs_repo_full | – | 58.2 | 45.8 (38.1–53.5) | 54.2 | 41.5 | 79.0 (73.0–84.5) |
| qwen2.5-coder:7b | P2 | zs_mschema | – | 64.3 | 48.4 (40.6–56.1) | 54.8 | 44.2 | 78.5 (72.5–84.5) |
| qwen2.5-coder:7b | P3 | zs_desc | – | 57.1 | 49.7 (41.9–57.4) | 55.7 | 45.0 | 78.5 (72.5–84.5) |
| qwen2.5-coder:7b | P4 | zs_rules | – | 59.2 | 49.7 (41.9–57.4) | 56.6 | 45.7 | 77.5 (71.5–83.0) |
| qwen2.5-coder:7b | P4m | ms_rules | – | 61.2 | 53.5 (45.8–61.3) | 59.5 | 49.3 | 77.5 (71.5–83.0) |
| qwen2.5-coder:7b | P5 | fs3 | – | 64.3 | 58.7 (50.3–65.8) | 61.4 | 52.9 | 83.5 (78.5–88.5) |
| qwen2.5-coder:7b | P5m | ms_fs3 | coder7b__evo-qwen7b__guarded | 60.2 | 58.7 (51.0–66.5) | 61.3 | 53.7 | 83.5 (78.5–88.5) |
| qwen2.5-coder:7b | P5m | ms_fs3 | – | 60.2 | 58.7 (51.0–66.5) | 61.3 | 53.5 | 83.5 (78.5–88.5) |
| qwen2.5-coder:7b | P6 | fs3_cot | – | 60.2 | 54.2 (46.5–61.9) | 61.8 | 50.3 | 76.0 (70.0–82.0) |
| qwen2.5-coder:3b | P1 | zs_repo | – | 37.8 | 35.5 (27.7–43.2) | 37.0 | 32.0 | 69.0 (62.0–75.5) |
| qwen2.5-coder:3b | P2 | zs_mschema | – | 40.8 | 36.1 (29.0–43.9) | 38.4 | 32.6 | 67.5 (61.0–74.0) |
| qwen2.5-coder:3b | P5m | ms_fs3 | – | 50.0 | 38.7 (31.0–46.5) | 40.4 | 35.0 | 76.5 (70.5–82.0) |
| qwen2.5-coder:1.5b | P1 | zs_repo | – | 24.5 | 21.9 (15.5–28.4) | 22.9 | 20.1 | 64.5 (58.0–71.0) |
| qwen2.5-coder:1.5b | P2 | zs_mschema | – | 28.6 | 23.9 (17.4–31.0) | 24.6 | 21.6 | 54.5 (47.0–61.5) |
| qwen2.5-coder:1.5b | P5m | ms_fs3 | – | 32.7 | 29.0 (21.9–36.1) | 28.3 | 26.2 | 66.0 (59.0–72.0) |
| qwen2.5:7b-instruct | P1 | zs_repo | – | 46.9 | 41.9 (34.2–49.7) | 47.2 | 37.6 | 72.0 (66.0–78.5) |
| qwen2.5:7b-instruct | P2 | zs_mschema | – | 42.9 | 43.9 (36.1–51.6) | 48.9 | 39.7 | 69.5 (63.0–76.0) |
| qwen2.5:7b-instruct | P5m | ms_fs3 | – | 54.1 | 54.2 (46.4–61.9) | 56.5 | 48.7 | 83.5 (78.0–88.5) |
<!-- /AUTO:RESULTS -->

### 5.2 Prompt ladder on Coder-7B: paired tests between neighbouring rungs

<!-- AUTO:PAIRED -->
| Set | Change | Gained | Lost | Net | McNemar p |
|---|---|---|---|---|---|
| bird_val | zs_ddl → zs_repo | 9 | 7 | +2 | 0.804 |
| bird_val | zs_repo → zs_repo_full | 7 | 5 | +2 | 0.774 |
| bird_val | zs_repo_full → zs_mschema | 8 | 2 | +6 | 0.109 |
| bird_val | zs_mschema → zs_desc | 0 | 7 | -7 | 0.016 |
| bird_val | zs_desc → zs_rules | 6 | 4 | +2 | 0.754 |
| bird_val | zs_rules → fs3 | 13 | 8 | +5 | 0.383 |
| bird_val | fs3 → fs3_cot | 9 | 13 | -4 | 0.523 |
| bird_val | zs_mschema → ms_rules | 3 | 6 | -3 | 0.508 |
| bird_val | ms_rules → ms_fs3 | 9 | 10 | -1 | 1.000 |
| bird_test | zs_ddl → zs_repo | 9 | 10 | -1 | 1.000 |
| bird_test | zs_repo → zs_repo_full | 4 | 5 | -1 | 1.000 |
| bird_test | zs_repo_full → zs_mschema | 10 | 6 | +4 | 0.454 |
| bird_test | zs_mschema → zs_desc | 5 | 3 | +2 | 0.727 |
| bird_test | zs_desc → zs_rules | 9 | 9 | +0 | 1.000 |
| bird_test | zs_rules → fs3 | 21 | 7 | +14 | 0.013 |
| bird_test | fs3 → fs3_cot | 10 | 17 | -7 | 0.248 |
| bird_test | zs_mschema → ms_rules | 13 | 5 | +8 | 0.096 |
| bird_test | ms_rules → ms_fs3 | 17 | 9 | +8 | 0.169 |
| spider_test | zs_ddl → zs_repo | 10 | 2 | +8 | 0.039 |
| spider_test | zs_repo → zs_repo_full | 0 | 0 | +0 | 1.000 |
| spider_test | zs_repo_full → zs_mschema | 6 | 7 | -1 | 1.000 |
| spider_test | zs_mschema → zs_desc | 0 | 0 | +0 | 1.000 |
| spider_test | zs_desc → zs_rules | 4 | 6 | -2 | 0.754 |
| spider_test | zs_rules → fs3 | 23 | 11 | +12 | 0.058 |
| spider_test | fs3 → fs3_cot | 7 | 22 | -15 | 0.008 |
| spider_test | zs_mschema → ms_rules | 4 | 6 | -2 | 0.754 |
| spider_test | ms_rules → ms_fs3 | 23 | 11 | +12 | 0.058 |
<!-- /AUTO:PAIRED -->

### 5.3 SkillEvo arms

<!-- AUTO:SKILLEVO -->
| Run | Evolver | BIRD test start | BIRD test final | Spider final | Edits | Kept | Kept negative | Library size |
|---|---|---|---|---|---|---|---|---|
| coder7b__evo-qwen7b__guarded | qwen2.5-coder:7b | 46.5 | 46.5 | 79.0 | 16.0 | 0.0 | 0.0 | 0 |
| coder7b__noskill | qwen2.5-coder:7b | 46.5 | 46.5 | 79.0 | nan | nan | nan | 0 |
<!-- /AUTO:SKILLEVO -->

## 6. Analysis

_Written from the numbers above; update as phases finish._

### 6.0 The story, with the reasons behind each result

**1. The baseline writes valid SQL that answers the wrong question.** Coder-7B with the repo prompt scores 46.5 on BIRD test and 79.0 on Spider test. Of its 83 wrong BIRD answers, 77 run and return the wrong rows. Six name a column that does not exist. That split told us where to look: a repair turn or a new schema layout can fix only those six, while a method that changes the model's choice of tables, filters and output columns can reach the other 77.

**2. Schema tweaks moved nothing we could measure.** We tried raw DDL, sample rows, an uncapped schema, M-Schema, column descriptions and output rules (P0 to P4). All score between 45.8 and 49.7 on BIRD test, and no step between neighbours passes a paired test. *Why:* the model already sees the tables it needs. Lifting the repo's 6,000-character cap (P1b) changed one answer. A cleaner layout presents the same facts, and the model picks the same wrong join or output column as before. Spider gave the one exception: two sample rows per table add 8 questions (p = 0.04). Spider's tables are small and cleanly named, and a sample value such as `'Asia'` shows the model how to spell the filter.

**3. Our 98-question selection set misled us once.** On val, column descriptions cost 7 questions (p = 0.016). On BIRD test they added 2. *Why:* on about 100 questions, a handful of flips reaches significance, and we chose the rungs to report on that same set, which inflates whatever looked best or worst there. We use val for decisions and quote test numbers in the report.

**4. Three retrieved examples gave the largest gain.** BM25 retrieves 3 solved questions from the train split and we add them to the prompt. BIRD test rises from 46.5 to 58.7: 26 questions gained, 7 lost, p = 0.001. On `challenging` questions accuracy doubles, from 21 to 42. *Why:* BIRD compares result rows, so the query has to follow the habits of the annotators who wrote the gold SQL. The examples show those habits, and the model copies them. In the questions it gained, Coder-7B:
- counted `T1.id` and joined through `sets`; zero-shot, it joined through `foreign_data` and counted every joined row;
- wrapped the result in `ROUND(..., 3)` when the question asked for three decimals;
- filtered on `label = '+'` and returned `molecule_id` alone; zero-shot, it skipped the filter and returned `label` as an extra column.

24 of the 26 gains turn a wrong-rows answer into a correct one. Answers with the gold number of output columns rise from 83% to 90%. The gained questions drew examples a little closer to their gold SQL (token overlap 0.62 against 0.56) and slightly fewer same-database examples (2.7 of 3 against 2.9). The model picks up query patterns from the examples. Familiarity with the database explains little, which eases part of the same-database concern in section 8.

**5. Step-by-step reasoning costs Coder-7B accuracy.** With a short reasoning step before the SQL, Spider drops 15 questions (p = 0.008) and BIRD drops 7, while output tokens triple. *Why:* after reasoning, the model writes its working into the query. It adds columns like `COUNT(*) AS Number_of_People`, joins tables the question never mentions, and concatenates first and last names. In 10 of the 22 Spider losses the column count differs from gold. Reasoning also flattens logic: for "nations that use English and Dutch" the model wrote `Language IN ('English','Dutch')`, which counts rows for either language. Qwen2.5-Coder learned to emit SQL directly, and at 7B the extra text adds drift and catches no errors.

**6. Model size decides BIRD; the prompt decides Spider.** Coder-3B with the best prompt scores 38.7 on BIRD test, below Coder-7B with the plain repo prompt at 46.5. *Why:* the dominant error changes with size. Coder-1.5B names a nonexistent column in 73 of 155 BIRD answers, Coder-3B in 39 and Coder-7B in 10. Matching question words to columns in BIRD's wide, messily named tables takes capacity that no prompt adds. Spider's small schemas ask much less, and Coder-3B with examples lands 2.5 points behind Coder-7B.

**7. M-Schema trips up Coder-1.5B.** On Spider it raises the 1.5B model's nonexistent-column errors from 20 to 48. The model attaches a column to the wrong alias: it writes `T2.Continent` when `Continent` sits in `country AS T1`. *Our reading:* public code is full of `CREATE TABLE` statements, so a small model has seen thousands of them and tracks which column belongs to which table. It has seen few M-Schema-style lists and loses the table boundaries.

**8. Qwen2.5-7B-Instruct closes the gap once it sees examples.** Zero-shot, Coder-7B leads it by 4.5 to 9 points. With examples the two tie on Spider at 83.5, and on BIRD Coder-7B leads by 7 questions (p = 0.21). The Instruct model gains 23 Spider questions from the examples; Coder-7B gains 9. *Our reading:* the Instruct model knows SQL and lacks the benchmark's habits, which the examples supply. Coder pretraining included public text-to-SQL data, so Coder-7B starts closer to those habits and has less left to learn.

**9. Self-evolution kept zero skills.** Coder-7B acted as both solver and evolver for 4 rounds and proposed 16 skills. The guard rejected all 16. We tested each proposal on 10 to 20 probe questions; across all 16 the skills fixed 1 answer and broke 4. *Why:*
- The skills restate advice the model follows already, such as "CAST before dividing" and "use strftime for text dates". Under greedy decoding, a paragraph the model agrees with leaves its SQL unchanged.
- The errors that matter are specific: which table to join through in this database, which column the annotator wanted. A solved example carries that detail. A general rule in prose has no room for it.
- The 7B evolver repeats itself. It proposed "percentages", "dates stored as text" and "date ranges" in every round under new titles, and the rejections did not change its next proposals.
- Retrieval matched 6 of the 16 skills to zero probe questions, so those six could change nothing.

Accepting all 16 would have cost 3 probe questions and lengthened every prompt.

**10. What we conclude.** For Qwen models from 1.5B to 7B on text-to-SQL, solved examples help and written guidance does not. Rules, descriptions, reasoning and evolved skills each move EX by less than the noise; 3 retrieved examples add 12 points on BIRD. That leads to three predictions for the pending runs:
- **LoRA on Coder-3B (D10/D11):** training on solved pairs puts the annotators' habits into the weights. We expect a gain close to the few-shot gain, with shorter prompts.
- **GLM-4.7-flash as evolver (C05):** if a stronger evolver also gets nothing accepted, skills as a format are the bottleneck. If it gets skills accepted, our 7B evolver was.
- **Self-consistency (A12):** voting fixes errors that vary between samples. Our model repeats the same misreading of a question, so we expect a small gain.

### 6.1 Prompt ladder on Coder-7B, `bird_val` only (preliminary, 2026-11-05)

These numbers come from the selection set (98 questions). Section 6.2 checks each claim on the test sets; three did not hold up (marked below).

| Rung | EX | Paired change vs parent |
|---|---|---|
| P0 raw DDL | 54.1 | |
| P1 repo prompt | 56.1 | +2 net (9 gained, 7 lost), p = 0.80 |
| P2 M-Schema | 64.3 | +8 net (11 / 3), p = 0.057 |
| P3 + descriptions | 57.1 | **-7 net (0 / 7), p = 0.016** |
| P4 + rules | 59.2 | +2 net, p = 0.75 |
| P5 + 3 few-shot | 64.3 | +5 net, p = 0.38 |
| P6 + CoT | 60.2 | -4 net, p = 0.52 |

What we read from it so far:

- **Schema format looked like the biggest lever.** On val, M-Schema (typed columns, 3 example values, FK lines) adds 8 points over the repo prompt, and lifting the cap alone (P1b) gives +2. **Did not replicate:** on BIRD test the format step gives +4 (p = 0.45) and the cap -1.
- **BIRD column descriptions hurt Coder-7B.** P3 gained no question and lost 7, the only significant change in the ladder. The losses come from small schemas too (`toxicology` grows from 465 to 716 prompt tokens), so length alone does not explain them. In the lost cases the model swaps `LIKE 'TR%_19'` for `= 'TR%_19'`, picks a different column or drops a subquery. Our reading: the value descriptions compete with the question's External knowledge, and a 7B model follows the wrong one. **Did not replicate:** on BIRD test, descriptions give +2 net (p = 0.73). We read the val drop as noise on 98 questions.
- **Rules and few-shot bring P5 back to P2's level on val.** We added a branch (P4m / P5m / P7m) that puts the rules, few-shot and voting on P2's schema instead.
- **Short CoT hurts this model** (-4 net, not significant). Qwen2.5-Coder writes better SQL when it answers directly.
- **On val, the repair turn helps zero-shot prompts only.** It fires on 8-16% of questions and adds 2-6 EX points to the zero-shot rungs (P2: 58.2 first attempt, 64.3 after repair). On val it adds nothing to P5/P6. **Corrected on test:** on BIRD test, repair adds about 4 points to every rung, few-shot included (P5: 54.8 first attempt, 58.7 after repair); on Spider it adds 0-1.5. After repair, 30-39 of the 98 answers run and return the wrong rows (`wrong_result`), against 5-8 `bad_column` errors. Further gains have to change which tables, filters and columns the model picks.

### 6.2 Prompt ladder on Coder-7B, test sets (2026-11-05)

| Rung | BIRD test EX | Spider test EX | Paired change vs parent (BIRD test / Spider) |
|---|---|---|---|
| P0 raw DDL | 47.1 | 75.0 | |
| P1 repo prompt | 46.5 | 79.0 | -1, p = 1.0 / **+8, p = 0.039** |
| P1b no 6k cap | 45.8 | 79.0 | -1, p = 1.0 / 0 |
| P2 M-Schema | 48.4 | 78.5 | +4, p = 0.45 / -1, p = 1.0 |
| P3 + descriptions | 49.7 | 78.5 | +2, p = 0.73 / 0 (Spider has none) |
| P4 + rules | 49.7 | 77.5 | 0 / -2 |
| P5 + 3 few-shot | **58.7** | **83.5** | **+14, p = 0.013** / +12, p = 0.058 |
| P6 + CoT | 54.2 | 76.0 | -7, p = 0.25 / **-15, p = 0.008** |
| P4m rules on P2 | 53.5 | 77.5 | +8 vs P2, p = 0.096 / -2 |
| P5m few-shot on P2 | **58.7** | **83.5** | +8 vs P4m, p = 0.17 / +12, p = 0.058 |

- **Few-shot retrieval is the one change that holds on every set.** Going from the repo prompt (P1) to P5m gains 26 BIRD test questions and loses 7: +19 net, p = 0.001, EX 46.5 [38.7, 54.2] to 58.7 [51.0, 66.5]. On Spider the same step is +9 net (p = 0.15). The gain concentrates on hard questions: BIRD `challenging` goes from 21.2 to 42.4.
- **P5 and P5m tie.** On BIRD test they differ on 10 questions, 5 each way, and they are identical on Spider. With demonstrations in the prompt, adding descriptions changes nothing. Pick P5m: its prompts are shorter (8.3 s vs 9.1 s per BIRD question).
- **Zero-shot schema tweaks are noise.** P0 through P4 all land between 45.8 and 49.7 on BIRD test, and no neighbouring step is significant there. The one significant zero-shot step is P0 to P1 on Spider (+8): two sample rows per table help Spider's small, cleanly named schemas.
- **CoT hurts, now with significance.** Spider loses 15 net (p = 0.008) and BIRD 7 net. CoT triples the output tokens (81 to 237 on BIRD) and doubles the latency. Soft-F1 moves from 61.4 to 61.8 on BIRD, so CoT answers land near the right rows and miss on exact columns or filters.

### 6.3 Model sweep (BIRD test / Spider test EX)

| Model | Repo prompt (P1) | M-Schema (P2) | Best prompt (P5m) | P1 to P5m, paired | s / BIRD question (P5m) |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 21.9 / 64.5 | 23.9 / 54.5 | 29.0 / 66.0 | +11 (p = 0.09) / +3 (p = 0.79) | 2.2 |
| Qwen2.5-Coder-3B | 35.5 / 69.0 | 36.1 / 67.5 | 38.7 / 76.5 | +5 (p = 0.54) / **+15 (p = 0.017)** | 4.1 |
| Qwen2.5-7B-Instruct | 41.9 / 72.0 | 43.9 / 69.5 | 54.2 / 83.5 | **+19 (p = 0.001) / +23 (p < 0.001)** | 8.4 |
| Qwen2.5-Coder-7B | 46.5 / 79.0 | 48.4 / 78.5 | **58.7 / 83.5** | **+19 (p = 0.001)** / +9 (p = 0.15) | 8.3 |

- **Size matters more than any prompt on BIRD.** Coder-3B with the best prompt (38.7) trails Coder-7B with the plain repo prompt (46.5), -12 net, p = 0.12. On Spider the gap closes: 3B with P5m (76.5) is within 2.5 points of 7B with P1.
- **Coder vs Instruct.** With zero-shot prompts the coder model leads by 4.5-9 points on both sets. With few-shot the general instruct model catches up: 54.2 vs 58.7 on BIRD (+7 net for Coder, p = 0.21) and a dead tie on Spider (83.5 each). The instruct model benefits most from demonstrations (+23 net on Spider).
- **M-Schema hurts the small models on Spider.** It costs Coder-1.5B 10 points and Coder-3B 1.5. Coder-1.5B attaches columns to the wrong table alias (section 6.0, point 7).
- **Latency doubles with each size step** (1.5B to 3B to 7B: 1.1, 2.2, 4.2 s per BIRD question), and the longer few-shot prompt doubles it again.

### 6.4 Self-evolving skills (Coder-7B as solver and evolver, guarded arm)

The guarded run went 4 rounds and proposed 16 skills, 4 per round. The guard rejected all 16: 10 flipped probe questions both ways for a net of -1 to +1, and 6 changed no answer ("inert"). The library ended empty, so the evolved-skill runs E01 and E02 reproduce their no-skill baselines exactly (58.7 / 83.5 and 46.5 / 79.0).

- The evolver repeats itself. "Calculate percentage", "dates stored as text" and "filter by date range" come back in rounds 1, 2, 3 and 4 under new names, so the evolver does not use the rejections.
- The proposed skills restate things the model already does (CAST before dividing, DISTINCT, JOIN on keys), which explains why so many are inert.
- Our reading: a 7B evolver cannot diagnose a 7B solver's errors. Run C05 (GLM-4.7-flash as evolver) to test that; it has not run yet.
- The guard kept 16 edits with no measurable benefit out of the library. Run the unguarded arm (C02, optional) to measure what accepting them costs.

### 6.5 Bottom line for the report

- **Best configuration so far:** Qwen2.5-Coder-7B with P5m (M-Schema + output rules + 3 BM25-retrieved demonstrations + one repair turn). BIRD test EX 58.7 [51.0, 66.5], Spider test EX 83.5 [78.5, 88.5], against 46.5 and 79.0 for the repo baseline.
- **What moved the numbers:** retrieved demonstrations (+12 EX on BIRD, significant) and model size. What did not: schema format, descriptions, rules, CoT (harmful) and self-evolved skills (none accepted).
- **Still pending:** LoRA on Coder-3B (D10/D11), GLM evolver (C05), self-consistency (A12). Run them with `caffeinate -i bash scripts/qwen_queue.sh D10 D11 C05 A12`.

### 6.6 Caveats specific to these results

- BIRD test has 155 questions, so a difference of 8 points sits inside the noise. Few comparisons survive a paired test: the few-shot step, CoT's loss on Spider, and the larger model-size gaps.
- The few-shot pool is the BIRD train split, on the same 11 databases as BIRD test (section 8). Same-database queries may account for part of the few-shot gain (section 6.0, point 4 suggests a small part); the official BIRD setting uses unseen databases.

<!-- ANALYSIS -->

## 7. How to reproduce

```bash
conda activate projects                      # python 3.10, mlx-lm installed
for m in qwen2.5-coder:7b qwen2.5-coder:3b qwen2.5-coder:1.5b qwen2.5:7b-instruct glm-4.7-flash; do ollama pull $m; done
python -m pytest -q tests                    # offline, ~2 s
python -m bench.plan list                    # every experiment + status
python -m bench.plan run A                   # prompt ladder on Coder-7B (~8 h incl. self-consistency)
python -m bench.plan run B                   # model sweep
python -m bench.plan run C01,C05             # SkillEvo arms
python -m bench.report                       # refresh Excel + this file
```

Raw data is not in git. This checkout links `data/raw` to the team copy under `Project/skillevo/data/raw` (excluded locally through `.git/info/exclude`). On a fresh clone, follow the download steps in `README.md`.

## 8. Threats to validity

- **Small sets.** 98 / 155 / 200 questions give EX intervals of about ±8-10 points. Read the paired McNemar columns before claiming a win.
- **Few-shot pool shares databases with BIRD test.** The BIRD pool is our train split: different questions on the same 11 databases. That is legitimate under our split, but it does not match the official BIRD setting, where dev databases are unseen. Zero-shot rows on `bird_all` are the leaderboard-comparable ones.
- **Quantisation.** Ollama Q4_K_M vs MLX 4-bit; only compare like with like.
- **R-VES timing noise.** We time 5 runs, in a single-threaded pass after each set; the official script times 100.
- **Cache keys.** The LLM cache key covers model, messages, max tokens and sampling settings, but not `num_ctx`. Keep `--num-ctx` at its default (16,384) for every bench run. For MLX runs the key is the `mlx:<label>` you type, so give every adapter its own label and restart `mlx_lm.server` when you switch adapters, or you will read cached answers from the previous model.
- **EX uses set semantics**, as the repo does, so row order and duplicates are ignored, which is slightly lenient for Spider `ORDER BY` questions.
