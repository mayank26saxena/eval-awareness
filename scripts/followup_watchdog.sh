#!/bin/bash
# Pod-side watchdog: when the pipeline driver is no longer running, wait 3 minutes (for log flush), then STOP this pod
# (not terminate: /workspace keeps every result). Uses the pod's own RUNPOD_API_KEY from the container environment.
cd /workspace/ea
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
while pgrep -f followup_driver.sh > /dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) watchdog: driver exited; stopping pod in 3 min" >> logs/followup/status
sleep 180
pgrep -f followup_driver.sh > /dev/null && { echo "$(date -u +%FT%TZ) watchdog: driver restarted, not stopping" >> logs/followup/status; exec bash $0; }
echo "$(date -u +%FT%TZ) watchdog: stopping pod $RUNPOD_POD_ID" >> logs/followup/status
runpodctl stop pod "$RUNPOD_POD_ID" >> logs/followup/status 2>&1
