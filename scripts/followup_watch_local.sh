#!/bin/bash
# Local watcher: print each new status line from the pod (one event per stage change), and generation progress every ~20 min.
# Exits when the pipeline reports it is done or the pod is unreachable for 10 minutes (e.g. the watchdog stopped it).
H="-o ConnectTimeout=20 -o BatchMode=yes ${POD_SSH:?set POD_SSH to the pod address, e.g. '-p PORT root@HOST'}"; seen=0; fails=0; tick=0
while true; do
  out=$(ssh $H 'cat /workspace/ea/logs/followup/status' < /dev/null 2>/dev/null) && fails=0 || { fails=$((fails+1)); [ $fails -ge 20 ] && { echo "pod unreachable for 10 min (stopped by watchdog?)"; exit 0; }; sleep 30; continue; }
  n=$(printf '%s\n' "$out" | grep -c .); if [ $n -gt $seen ]; then printf '%s\n' "$out" | tail -n $((n-seen)); seen=$n; fi
  printf '%s\n' "$out" | grep -qE "ALL DONE|stopping pod|SMOKE FAILED" && exit 0
  tick=$((tick+1)); if [ $((tick % 40)) -eq 0 ]; then ssh $H 'tail -1 $(ls -t /workspace/ea/logs/followup/gen_*.log /workspace/ea/logs/followup/judge_*.log 2>/dev/null | head -1) | cut -c1-120' < /dev/null 2>/dev/null; fi
  sleep 30
done
