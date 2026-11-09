## EX % per run

| Model | Rung | Strategy | bird_val | spider_val | bird_test | spider_test | BIRD test Soft-F1 | BIRD test R-VES | ctx overflow | s/q (BIRD val) |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b-it-q4_K_M | P1 | zs_repo | 19.4 | – | 18.1 | 33.5 | 19.4 | – | 0.0 | 5.0 |
| gemma4:e2b-it-q4_K_M | S1 | sm_extract | 40.8 | – | 42.6 | 74.0 | 45.5 | – | 0.0 | 3.2 |
| gemma4:e2b-it-q4_K_M | S6 | sm_pruned_fs3 | 46.9 | – | 45.2 | 65.5 | 47.6 | – | 0.0 | 3.2 |

## Paired tests (same questions)

| Model | Change | Kind | Set | Gained | Lost | Net | McNemar p |
|---|---|---|---|---|---|---|---|
| gemma4:e2b-it-q4_K_M | zs_repo → sm_extract | vs baseline | bird_val | 21 | 0 | +21 | 0.000 |
| gemma4:e2b-it-q4_K_M | zs_repo → sm_extract | vs baseline | bird_test | 38 | 0 | +38 | 0.000 |
| gemma4:e2b-it-q4_K_M | zs_repo → sm_extract | vs baseline | spider_test | 81 | 0 | +81 | 0.000 |
| gemma4:e2b-it-q4_K_M | zs_repo → sm_pruned_fs3 | vs baseline | bird_val | 31 | 4 | +27 | 0.000 |
| gemma4:e2b-it-q4_K_M | zs_repo → sm_pruned_fs3 | vs baseline | bird_test | 50 | 8 | +42 | 0.000 |
| gemma4:e2b-it-q4_K_M | zs_repo → sm_pruned_fs3 | vs baseline | spider_test | 75 | 11 | +64 | 0.000 |

## Test-set breakdown

| Run | Set | EX | Valid SQL | Simple | Moderate | Challenging | Errors |
|---|---|---|---|---|---|---|---|
| gemma4-e2b-it-q4-K-M__sm_extract | bird_test | 42.6 | 88.4 | 61.4 | 38.5 | 27.3 | wrong_result 71, bad_column 13, syntax 3, timeout 1, bad_function 1 |
| gemma4-e2b-it-q4-K-M__sm_extract | spider_test | 74.0 | 97.0 | 79.8 | 70.3 | 54.5 | wrong_result 46, bad_column 4, other_error 2 |
| gemma4-e2b-it-q4-K-M__sm_pruned_fs3 | bird_test | 45.2 | 96.8 | 59.1 | 41.0 | 36.4 | wrong_result 80, bad_column 3, syntax 1, other_error 1 |
| gemma4-e2b-it-q4-K-M__sm_pruned_fs3 | spider_test | 65.5 | 99.0 | 69.3 | 65.6 | 45.5 | wrong_result 67, bad_column 1, other_error 1 |
| gemma4-e2b-it-q4-K-M__zs_repo | bird_test | 18.1 | 34.8 | 34.1 | 14.1 | 6.1 | syntax 99, wrong_result 26, bad_column 2 |
| gemma4-e2b-it-q4-K-M__zs_repo | spider_test | 33.5 | 42.0 | 47.4 | 15.6 | 13.6 | syntax 114, wrong_result 17, other_error 2 |
