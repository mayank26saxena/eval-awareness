#!/bin/bash
# Pod-side watchdog: when neither the driver nor the job queue is running, wait 3 minutes, check again, then STOP this
# pod (not terminate: /workspace keeps every result). Uses the pod's own RUNPOD_API_KEY from the container environment.
cd /workspace/ea
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
busy() { pgrep -f "followup_(driver|queue)\.sh" > /dev/null; }
while true; do
  while busy; do sleep 60; done
  echo "$(date -u +%FT%TZ) watchdog: pipeline idle; stopping pod in 3 min unless work resumes" >> logs/followup/status
  sleep 180
  busy && continue
  echo "$(date -u +%FT%TZ) watchdog: stopping pod $RUNPOD_POD_ID" >> logs/followup/status
  runpodctl stop pod "$RUNPOD_POD_ID" >> logs/followup/status 2>&1
  exit 0
done
