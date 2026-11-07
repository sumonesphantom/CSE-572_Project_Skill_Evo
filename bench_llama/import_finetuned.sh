#!/bin/zsh
# After LoRA training: pick the checkpoint with the lowest validation loss, fuse it into the
# base model, import into Ollama as llama3.2-1b-sql at Q8_0 (same precision as the baseline
# llama3.2:1b, with the baseline's chat template), then evaluate it exactly like the baseline.
set -e
cd "$(dirname "$0")/.."
OLLAMA=${OLLAMA:-ollama}   # path to the ollama binary
caffeinate -i -w $$ &
[ -n "$1" ] && while kill -0 $1 2>/dev/null; do sleep 15; done

AD=adapters/llama1b-sql-lora
best=$(sed 's/\x1b\[[0-9;]*m//g' logs/train.log | awk '$2=="val"{print $1, $3}' | awk '$1>1' | sort -k2 -g | head -1 | cut -d' ' -f1)
echo "best checkpoint by validation loss: step $best"
mkdir -p adapters/chosen && cp $AD/adapter_config.json adapters/chosen/
if [ "$best" = "1000" ]; then cp $AD/adapters.safetensors adapters/chosen/; else cp $AD/$(printf "%07d" $best)_adapters.safetensors adapters/chosen/adapters.safetensors; fi

.venv-mlx/bin/mlx_lm.fuse --model models/llama-3.2-1b-instruct-bf16 --adapter-path adapters/chosen \
  --save-path models/llama1b-sql-fused --dequantize 2>&1 | tail -2

# Modelfile: fused weights + the baseline model's own chat template/stop tokens
$OLLAMA show llama3.2:1b --modelfile | awk '/^TEMPLATE/{f=1} f{print} f&&/"""$/&&NR>1&&!/^TEMPLATE/{exit}' > /tmp/llama-template.txt
{ echo "FROM $(pwd)/models/llama1b-sql-fused"; cat /tmp/llama-template.txt;
  echo 'PARAMETER stop "<|start_header_id|>"'; echo 'PARAMETER stop "<|end_header_id|>"'; echo 'PARAMETER stop "<|eot_id|>"'; } > models/Modelfile.llama1b-sql
$OLLAMA create llama3.2-1b-sql -f models/Modelfile.llama1b-sql --quantize int8
$OLLAMA show llama3.2-1b-sql | head -8

# evaluate: single attempt (pruned prompt) and the 5-sample vote, on val + test sets
.venv/bin/python -m bench_llama.evaluate --model llama3.2-1b-sql --variants pruned,sc5_nofs \
  --sets val,spider_val,bird_test,spider_transfer
echo "all done $(date)"
