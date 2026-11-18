#!/bin/bash
# Job queue that runs after the driver: executes logs/followup/jobs/*.sh one at a time, in name order, never while the
# driver is running (one GPU). New jobs can be dropped in at any time. Exits after 10 idle minutes with the driver gone.
cd /workspace/ea; J=logs/followup/jobs; mkdir -p $J/done; idle=0
say() { echo "$(date -u +%FT%TZ) $*" >> logs/followup/status; }
while true; do
  if pgrep -f "followup_driver\.sh" > /dev/null; then idle=0; sleep 60; continue; fi
  j=$(ls $J/*.sh 2>/dev/null | sort | head -1)
  if [ -n "$j" ]; then
    idle=0; b=$(basename $j .sh); say "job $b started"
    bash $j > logs/followup/job_$b.log 2>&1 && say "job $b finished" || say "job $b FAILED (see logs/followup/job_$b.log)"
    mv $j $J/done/
  else
    idle=$((idle+1)); [ $idle -ge 10 ] && { say "queue empty, exiting"; exit 0; }; sleep 60
  fi
done
