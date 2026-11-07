#!/bin/zsh
# Wait for a running eval (pid $1) to finish, free Ollama's memory, then LoRA fine-tune.
# caffeinate keeps the Mac awake for the whole wait + training.
cd "$(dirname "$0")/.."
OLLAMA=${OLLAMA:-ollama}   # path to the ollama binary
caffeinate -i -w $$ &
while kill -0 $1 2>/dev/null; do sleep 15; done
$OLLAMA stop llama3.2:1b 2>/dev/null
echo "training started $(date)"
# 1000 iters x batch 4 = 4,000 examples (~half an epoch); eval + checkpoint every 200 steps
.venv-mlx/bin/mlx_lm.lora --model models/llama-3.2-1b-instruct-bf16 --train --data data/finetune \
  --fine-tune-type lora --mask-prompt --num-layers 16 --batch-size 4 --grad-checkpoint \
  --iters 1000 --learning-rate 1e-4 --max-seq-length 1024 \
  --steps-per-report 20 --steps-per-eval 200 --val-batches 25 --save-every 200 \
  --adapter-path adapters/llama1b-sql-lora --seed 0
echo "training finished $(date)"
