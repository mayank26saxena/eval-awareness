#!/usr/bin/env bash
# Run the full pipeline for one model and one dataset version.
#
# On macOS each step runs under `caffeinate -ims`, so idle/system sleep is blocked
# for as long as the step is running. A closed lid still sleeps unless the machine
# is on power with an external display.
#
# Usage:
#   scripts/run_pipeline.sh <model> <dataset-stem> [steps...]
#   scripts/run_pipeline.sh Qwen/Qwen2.5-7B-Instruct v3_main generate judge probe transfer
#
# Steps: generate | judge | probe | transfer   (default: all four, in that order)
# Logs:  logs/<model-slug>_<dataset>_<step>.log

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; DATA="$2"; shift 2
if [[ $# -eq 0 ]]; then STEPS=(generate judge probe transfer); else STEPS=("$@"); fi

PY=".venv/bin/python"
SLUG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
mkdir -p logs

if command -v caffeinate >/dev/null 2>&1; then KEEP=(caffeinate -ims); else KEEP=(); fi
export PYTHONUNBUFFERED=1

run() {
  local step="$1"; shift
  local log="logs/${SLUG}_${DATA}_${step}.log"
  echo "[$(date '+%F %T')] ${step} -> ${log}"
  "${KEEP[@]}" "$PY" "$@" > "$log" 2>&1
}

for step in "${STEPS[@]}"; do
  case "$step" in
    generate) run generate scripts/generate.py --model "$MODEL" --data "data/rendered/${DATA}.jsonl" --max-new-tokens 300 ;;
    judge)    run judge    scripts/judge.py "results/${SLUG}/responses_${DATA}.jsonl" --data "data/rendered/${DATA}.jsonl" ;;
    probe)    run probe    scripts/probe.py --model "$MODEL" --data "data/rendered/${DATA}.jsonl" ;;
    transfer) run transfer scripts/probe_transfer.py --model "$MODEL" --train "data/rendered/${DATA}.jsonl" ;;
    *) echo "unknown step: $step" >&2; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] done"
