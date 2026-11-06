#!/usr/bin/env bash
# Unattended LoRA step for one Qwen size: build SFT data, train (lora mode), serve with mlx_lm.server,
# evaluate with bench.run, stop the server. "base" mode evaluates the MLX base model (control).
#   bash scripts/finetune_auto.sh coder3b base     # control, ~30 min
#   bash scripts/finetune_auto.sh coder3b lora     # train + eval, ~2 h
#   SMOKE=1 bash scripts/finetune_auto.sh coder1.5b lora   # 20 training iters, 5 eval questions (~5 min)
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=${1:?model key: coder1.5b|coder3b|coder7b}
MODE=${2:?base|lora}
BIN=/opt/anaconda3/envs/projects/bin
PY=$BIN/python
PORT=${PORT:-8080}
CFG=finetune/configs/${KEY}_r16.yaml
HF=$(awk '/^model:/{print $2}' "$CFG")
ADAPTER=$(awk '/^adapter_path:/{print $2}' "$CFG")
SETS=bird_val,bird_test,spider_test
LIMIT=""
if [ "${SMOKE:-0}" = 1 ]; then LIMIT="--limit 5"; fi

# free unified memory: unload Ollama models before MLX loads its own copy
for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop "$m" || true; done

if [ "$MODE" = lora ]; then
  if [ ! -f finetune/data/ms_rules/train.jsonl ]; then
    $PY finetune/build_sft.py --strategy ms_rules --spider-max 3000 --max-tokens 4000 --out finetune/data/ms_rules
  fi
  if [ "${SMOKE:-0}" = 1 ]; then
    ADAPTER=${ADAPTER}-smoke
    $BIN/mlx_lm.lora -c "$CFG" --iters 20 --adapter-path "$ADAPTER" --steps-per-eval 10 --val-batches 2
  elif [ ! -f "$ADAPTER/adapters.safetensors" ]; then
    $BIN/mlx_lm.lora -c "$CFG"
  fi
  LABEL=mlx:$(basename "$ADAPTER")
  SERVE=(--model "$HF" --adapter-path "$ADAPTER")
else
  LABEL=mlx:${KEY}-base
  SERVE=(--model "$HF")
fi

$BIN/mlx_lm.server "${SERVE[@]}" --port "$PORT" > "runs/logs/mlx_server_${KEY}_${MODE}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break
  sleep 2
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "mlx_lm.server did not start (see runs/logs)"; exit 1; }

# one worker: mlx_lm.server handles one request at a time
$PY -m bench.run --model "$LABEL@localhost:$PORT" --rungs "${RUNGS:-ms_rules}" --sets $SETS --workers 1 $LIMIT
