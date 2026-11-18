#!/bin/bash
# Usage: followup_run.sh <split: pilot|main> <k> <model> [<model> ...]   (generation only, resumable)
cd /workspace/ea && . /workspace/venv/bin/activate && export HF_HOME=/workspace/hf VLLM_LOGGING_LEVEL=WARNING
SPLIT=$1; K=$2; shift 2; mkdir -p logs/followup
for M in "$@"; do
  S=$(basename $M | tr A-Z a-z)
  python scripts/followup_generate.py --model $M --data data/followup/rendered/$SPLIT.jsonl \
     --out results/followup/$SPLIT/$S.jsonl --k $K > logs/followup/gen_${SPLIT}_$S.log 2>&1 \
     && echo "$(date -u +%FT%TZ) $SPLIT $S generated" >> logs/followup/status \
     || echo "$(date -u +%FT%TZ) $SPLIT $S FAILED gen" >> logs/followup/status
done
