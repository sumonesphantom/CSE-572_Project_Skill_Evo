#!/usr/bin/env bash
# Reduced queue for the Qwen benchmark (~19 h), most important results first, so stopping early
# still leaves a complete story. Every step resumes from cache / preds.jsonl; Ctrl-C and rerun any time.
#   caffeinate -i bash scripts/qwen_queue.sh          # everything below
#   bash scripts/qwen_queue.sh B C01                  # only some plan ids / phases
#   EXTRAS=1 bash scripts/qwen_queue.sh               # also the optional runs at the end
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/opt/anaconda3/envs/projects/bin/python}
mkdir -p runs/logs
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(
    A02,A09,A03,A04,A05,A06   # 1. finish the Coder-7B ladder on test sets               ~2.5 h  (cum.  2.5)
    A10,A11                   # 2. P2-based branch: rules / few-shot, no descriptions    ~1.0 h  (cum.  3.5)
    B00,B01,B02               # 3. Coder-3B:   repo prompt, M-Schema, best few-shot      ~0.8 h
    B10,B11,B12               #    Coder-1.5B                                            ~0.6 h
    B20,B21,B22               #    Qwen2.5-7B-Instruct                                   ~1.2 h  (cum.  6.1)
    C10 C01                   # 4. SkillEvo: no-skill + guarded, Qwen-7B as evolver      ~3.1 h  (cum.  9.2)
    E02,E01                   # 5. evolved skills under repo prompt / best prompt        ~0.9 h  (cum. 10.1)
    D10 D11                   # 6. LoRA on Coder-3B (MLX base control, then adapter)     ~2.5 h  (cum. 12.6)
    C05                       # 7. SkillEvo reference: GLM-4.7-flash as evolver          ~4.0 h  (cum. 16.6)
    A12                       # 8. self-consistency (5 votes) on the best prompt         ~1.6 h  (cum. 18.2)
  )
  if [ "${EXTRAS:-0}" = 1 ]; then
    STEPS+=(C02 D20 D21 C07 B03,B13,B23 A07)   # unguarded drift, LoRA 1.5B, 3B self-evolution, ...
  fi
fi
for s in "${STEPS[@]}"; do
  echo "=== $(date '+%F %T') step $s" | tee -a runs/logs/queue.log
  "$PY" -m bench.plan run "$s" >> "runs/logs/queue_${s//,/_}.log" 2>&1 || echo "step $s failed (see runs/logs/queue_${s//,/_}.log)" | tee -a runs/logs/queue.log
  "$PY" -m bench.report >> runs/logs/queue.log 2>&1
done
echo "=== $(date '+%F %T') queue finished" | tee -a runs/logs/queue.log
