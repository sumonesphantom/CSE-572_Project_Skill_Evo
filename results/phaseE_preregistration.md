# Phase E pre-registration (written before any Phase E test-set run)

Date: 2026-11-08. Test sets opened once, for exactly these two configurations, chosen on bird_val only:

| ID | Model (Ollama) | Strategy | Why | bird_val EX |
|---|---|---|---|---|
| E1 | phi4-mini (official Q4_K_M) | b1_fs3 | best prompt-only setup (Phase B winner, +6/-0 vs a4_fs3, p=0.031) | 34.7 |
| E2 | phi4mini-d1 (LoRA d1, Q4_K_M) | b1_fs3 | best overall on bird_val (Phase D; +7 net vs base, p=0.21) | 37.8 |

Sets: bird_test (155, repo split, includes the duplicated bird-137 twice as in Venkat's tables) and
spider_test (seeded 200 of spider:test, same sample as config.yaml). Metrics: EX with bootstrap 95% CI,
Soft-F1, R-VES (BIRD), paired McNemar E1 vs E2 and vs Phase A a4_fs3 on the same questions.
No further tuning after these runs. Already-run test numbers (Phase A rungs a1-a4) stay as reported.
