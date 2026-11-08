#!/usr/bin/env bash
# Phase C: SkillEvo with Phi-4-mini as solver and evolver, three arms in order (cache shared, so the
# no-skill answers are reused by the monitor's baseline in the guarded arms).
#   caffeinate -i bash scripts/phi_skillevo.sh            # all three arms (~10 h)
#   bash scripts/phi_skillevo.sh guarded_fix               # one arm
# A finished arm is skipped. An arm that stopped half way must be removed first (rm -r runs/<name>);
# rerunning it is fast because every model answer is cached.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p runs/logs
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(noskill guarded guarded_fix)
for a in "${ARMS[@]}"; do
  case "$a" in
    noskill)     arm=noskill; cfg=configs/phi/phi4mini__base.yaml ;;
    guarded)     arm=guarded; cfg=configs/phi/phi4mini__base.yaml ;;
    guarded_fix) arm=guarded; cfg=configs/phi/phi4mini__fix.yaml ;;
    *) echo "unknown arm $a (noskill | guarded | guarded_fix)"; exit 1 ;;
  esac
  name="phi4mini__$a"; dir="runs/$name"
  if [ -f "$dir/rounds.jsonl" ]; then
    last=$("$PY" -c "import json;print(max(json.loads(l)['round'] for l in open('$dir/rounds.jsonl') if l.strip()))")
    want=$([ "$arm" = noskill ] && echo 0 || echo 4)
    if [ "$last" -ge "$want" ]; then echo "== $name already finished, skipping"; continue; fi
    echo "== $name stopped at round $last: remove $dir and rerun (cached answers make it fast)"; exit 1
  fi
  echo "=== $(date '+%F %T') $name" | tee -a runs/logs/phaseC.log
  "$PY" run.py --arm "$arm" --config "$cfg" --name "$name" 2>&1 | tee -a runs/logs/phaseC.log
done
"$PY" analyze.py runs/phi4mini__noskill runs/phi4mini__guarded runs/phi4mini__guarded_fix 2>&1 | tee -a runs/logs/phaseC.log
