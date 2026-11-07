#!/usr/bin/env bash
# Phase D: LoRA fine-tuning of Phi-4-mini with MLX, then evaluation through phibench. Free/local only.
#
#   bash scripts/phi_lora.sh setup            # install mlx-lm, download one SynSQL-2.5M shard (~110 MB)
#   bash scripts/phi_lora.sh data             # build finetune/data/{d1,d2}
#   bash scripts/phi_lora.sh probe            # which example lengths / LoRA depths fit in this Mac's memory
#   bash scripts/phi_lora.sh smoke            # 20 training iters + 5 eval questions: measures speed and memory
#   bash scripts/phi_lora.sh train d1 [ITERS] # train adapter finetune/adapters/d1 (default 400 iters)
#   bash scripts/phi_lora.sh eval base        # D0: MLX base model, no adapter (the control)
#   bash scripts/phi_lora.sh eval d1          # evaluate adapter d1 on bird_val (b1_rules, b1_fs3)
#
# Run long steps under caffeinate and on mains power:  caffeinate -i bash scripts/phi_lora.sh train d1
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
BASE=mlx-community/Phi-4-mini-instruct-4bit
CFG=finetune/phi4mini_lora.yaml
PORT=${PORT:-8080}
SETS=${SETS:-bird_val}
RUNGS=${RUNGS:-b1_rules,b1_fs3}
SYN_REPO=${SYN_REPO:-Abdulvajid/SynSQL-2.5M}
mkdir -p runs/logs

free_memory() {  # Ollama keeps models in unified memory; MLX needs that room
  for m in $(ollama ps 2>/dev/null | awk 'NR>1{print $1}'); do ollama stop "$m" || true; done
}

serve_and_eval() {  # $1 label, $2 adapter dir or "", $3 extra phibench args
  local label=$1 adapter=$2 extra=${3:-}
  local args=(--model "$BASE" --port "$PORT")
  [ -n "$adapter" ] && args+=(--adapter-path "$adapter")
  free_memory
  "$PY" -m mlx_lm server "${args[@]}" > "runs/logs/mlx_server_${label}.log" 2>&1 &
  local pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 90); do curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break; sleep 2; done
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "mlx_lm.server did not start, see runs/logs/mlx_server_${label}.log"; exit 1; }
  # one worker: mlx_lm.server answers one request at a time
  "$PY" -m phibench.run --backend openai --base-url "http://localhost:$PORT" --model default_model \
    --label "phi4mini-mlx-$label" --strategies "$RUNGS" --sets "$SETS" --workers 1 $extra
  kill $pid 2>/dev/null || true
  trap - EXIT
}

case "${1:-}" in
  setup)
    "$PY" -m pip install -q "mlx-lm[train]>=0.28" pyarrow huggingface_hub
    "$PY" -c "import mlx_lm, platform; print('mlx-lm', mlx_lm.__version__, platform.machine())"
    "$PY" -c "from huggingface_hub import snapshot_download as d; d('$SYN_REPO', repo_type='dataset', allow_patterns=['data/train-00000-*'], local_dir='data/raw/synsql')"
    ls -lh data/raw/synsql/data/
    ;;
  data)
    "$PY" -m phibench.sft --mix d1 --out finetune/data/d1 --max-tokens 8000
    "$PY" -m phibench.sft --mix d2 --out finetune/data/d2 --max-tokens 8000 --synsql data/raw/synsql/data/train-00000-*.parquet
    ;;
  probe)
    free_memory
    echo "RAM: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB   GPU wired limit: $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo default) MB"
    "$PY" - <<'PYEOF'
import json, pathlib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
rows = [json.loads(l) for l in open("finetune/data/d1/train.jsonl")]
def n(r):
    ids = tok.apply_chat_template(r["messages"], tokenize=True)
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
scored = sorted(((n(r), r) for r in rows), key=lambda x: x[0])
for target in (2000, 3000, 4000, 5500, 7000):
    k, r = min(scored, key=lambda x: abs(x[0] - target))
    d = pathlib.Path(f"finetune/data/probe_{target}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.jsonl").write_text((json.dumps(r) + "\n") * 4)
    (d / "valid.jsonl").write_text(json.dumps(r) + "\n")
    print(f"probe_{target}: example of {k} tokens")
PYEOF
    for layers in 16 8; do
      for target in 2000 3000 4000 5500 7000; do
        printf "layers=%-2s target=%-5s " "$layers" "$target"
        if "$PY" -m mlx_lm lora -c "$CFG" --data "finetune/data/probe_$target" --adapter-path finetune/adapters/probe \
             --num-layers "$layers" --iters 2 --steps-per-report 1 --steps-per-eval 1000 --val-batches 1 \
             --save-every 1000 > "runs/logs/probe_${layers}_${target}.log" 2>&1; then
          echo "OK   $(grep -io 'peak[^|]*' "runs/logs/probe_${layers}_${target}.log" | tail -1)"
        else
          echo "FAIL $(grep -o 'Insufficient Memory' "runs/logs/probe_${layers}_${target}.log" | head -1)"
          break
        fi
      done
    done
    ;;
  smoke)
    free_memory
    # the 20 longest training examples: tests peak memory before a multi-hour run
    "$PY" - <<'PYEOF'
import json, pathlib
src, dst = pathlib.Path("finetune/data/d1"), pathlib.Path("finetune/data/smoke")
dst.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(src / "train.jsonl")]
rows.sort(key=lambda r: -sum(len(m["content"]) for m in r["messages"]))
(dst / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:20]))
(dst / "valid.jsonl").write_text("".join(list(open(src / "valid.jsonl"))[:4]))
print("smoke data: longest example", sum(len(m["content"]) for m in rows[0]["messages"]), "chars")
PYEOF
    "$PY" -m mlx_lm lora -c "$CFG" --data finetune/data/smoke --adapter-path finetune/adapters/smoke \
      --iters 20 --steps-per-report 5 --steps-per-eval 20 --val-batches 2 --save-every 20 2>&1 | tee runs/logs/lora_smoke.log
    RUNGS=b1_rules serve_and_eval smoke finetune/adapters/smoke "--limit 5 --tag smoke --no-ves"
    ;;
  train)
    mix=${2:?mix: d1 or d2}; iters=${3:-400}
    free_memory
    "$PY" -m mlx_lm lora -c "$CFG" --data "finetune/data/$mix" --adapter-path "finetune/adapters/$mix" \
      --iters "$iters" 2>&1 | tee "runs/logs/lora_${mix}.log"
    ;;
  eval)
    name=${2:?eval: base, d1 or d2}
    if [ "$name" = base ]; then serve_and_eval base ""; else serve_and_eval "$name" "finetune/adapters/$name"; fi
    ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
