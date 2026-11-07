#!/bin/zsh
# Run a bench command in the background and keep the Mac awake exactly as long as it runs.
#   bench_llama/run_bg.sh <logfile> <args for bench.evaluate...>
cd "$(dirname "$0")/.."
log=$1; shift
nohup .venv/bin/python -m bench_llama.evaluate "$@" > "$log" 2>&1 &
pid=$!
nohup caffeinate -i -w $pid > /dev/null 2>&1 &
echo "started pid $pid (caffeinate holds the Mac awake until it exits)"
