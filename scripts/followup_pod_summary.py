"""Runs on the pod (stdlib only). Prints one JSON snapshot of the follow-up pipeline for the local tracker.

  python3 scripts/followup_pod_summary.py                    # snapshot
  python3 scripts/followup_pod_summary.py examples <split> <model-slug> [n_items]   # replies for a few items, every cell
"""
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

ROOT = "/workspace/ea"
CELLS = [f"C{i}" for i in range(12)]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def read_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line being written right now
    return out


def log_progress(p):
    text = open(p, errors="replace").read()
    d = dict(log=os.path.basename(p), mtime=int(os.path.getmtime(p)))
    m = re.findall(r"(\d+) prompts x (\d+) samples; (\d+) done; (\d+) to do", text)
    if m:
        d.update(prompts=int(m[-1][0]), k=int(m[-1][1]), done_before=int(m[-1][2]), todo=int(m[-1][3]))
    m = re.findall(r"^(\d+)/(\d+)\s+(\d+)s\s+(\d+) tok/s", text, re.M)
    if m:
        d.update(progress=int(m[-1][0]), of=int(m[-1][1]), secs=int(m[-1][2]), tok_s=int(m[-1][3]))
    m = re.findall(r"(\d+) replies; (\d+) judged; (\d+) to judge", text)
    if m:
        d.update(replies=int(m[-1][0]), judged_before=int(m[-1][1]), to_judge=int(m[-1][2]))
    m = re.findall(r"judged (\d+) in (\d+)s", text)
    if m:
        d.update(judged=int(m[-1][0]), judge_secs=int(m[-1][1]))
    if "Traceback" in text:
        d["error"] = [l for l in text.splitlines() if l.strip()][-1][:300]
    return d


def file_summary(p):
    rows = read_jsonl(p)
    d = dict(file=os.path.relpath(p, ROOT), n=len(rows), mtime=int(os.path.getmtime(p)))
    by = {}
    for r in rows:
        by.setdefault(r["cell"], 0)
        by[r["cell"]] += 1
    d["by_cell"] = by
    d["items"] = len({r["task_id"] for r in rows})
    if os.path.basename(p).startswith("judged"):
        prot = {}
        for r in rows:
            key = f"{r['benchmark']}|{r['cell']}"
            s = prot.setdefault(key, [0, 0, 0])  # good, scored, unparsed
            if r.get("good") is None:
                s[2] += 1
            else:
                s[0] += int(r["good"])
                s[1] += 1
        d["protection"] = prot
        d["unparsed"] = sum(v[2] for v in prot.values())
    return d


def snapshot():
    os.chdir(ROOT)
    st = open("logs/followup/status").read().splitlines() if os.path.exists("logs/followup/status") else []
    procs = [l for l in sh("ps -eo pid,etimes,args | grep -E 'followup_(generate|judge|steer|recognition)' | grep -v grep").splitlines()]
    running = []
    for l in procs:
        parts = l.split(None, 2)
        args = parts[2]
        m = re.search(r"--(?:model|judge) (\S+)", args)
        running.append(dict(pid=int(parts[0]), secs=int(parts[1]),
                            kind="judge" if "followup_judge" in args else "steer" if "followup_steer" in args else "recognition" if "followup_recognition" in args else "generate",
                            model=m.group(1) if m else "", split=("steer" if "steer/" in args else (re.search(r"rendered/(\w+)\.jsonl", args) or [None, ""])[1])))
    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits").split(",")
    files = sorted(glob.glob("results/followup/pilot/*.jsonl") + glob.glob("results/followup/main/*.jsonl") + glob.glob("results/followup/steer/*.jsonl"))
    return dict(
        now=int(time.time()), status=st, running=running,
        driver=bool(sh("pgrep -f followup_driver.sh")), watchdog=bool(sh("pgrep -f followup_watchdog.sh")),
        gpu=dict(util=int(gpu[0]), mem=int(gpu[1]), total=int(gpu[2])) if len(gpu) == 3 else None,
        disk=sh("df -h /workspace | tail -1 | awk '{print $3\" used of \"$2}'"),
        hf_cache=sh("du -sh /workspace/hf 2>/dev/null | cut -f1"),
        logs=[log_progress(p) for p in sorted(glob.glob("logs/followup/*.log"), key=os.path.getmtime)
              if re.search(r"(gen|judge|steer|recog)", os.path.basename(p))],
        files=[file_summary(p) for p in files],
    )


def examples(split, slug, n=3):
    os.chdir(ROOT)
    judged = f"results/followup/{split}/judged_{slug}.jsonl"
    src = judged if os.path.exists(judged) else f"results/followup/{split}/{slug}.jsonl"
    rows = [r for r in read_jsonl(src) if r["k"] == 0]
    prompts = {r["id"]: r for r in read_jsonl(f"data/followup/rendered/{split}.jsonl")}
    by_item = {}
    for r in rows:
        by_item.setdefault(r["task_id"], {})[r["cell"]] = r
    full = [t for t, v in by_item.items() if len(v) >= 10]
    random.seed(int(time.time() // 600))  # a new draw every 10 minutes
    pick = random.sample(full, min(n, len(full)))
    out = []
    for t in pick:
        cells = []
        for c in CELLS:
            r = by_item[t].get(c)
            if not r:
                continue
            p = prompts.get(r["id"], {})
            msgs = p.get("messages", [{}, {}])
            cells.append(dict(cell=c, response=r["response"][:1200], good=r.get("good"), finish=r.get("finish"),
                              system_tail=msgs[0].get("content", "")[-260:], user_head=msgs[1].get("content", "")[:260],
                              user_tail=msgs[1].get("content", "")[-200:]))
        out.append(dict(task_id=t, benchmark=next(iter(by_item[t].values()))["benchmark"],
                        hidden_fact=next(iter(prompts.values()), {}).get("hidden_fact") if False else
                        prompts.get(f"{t}__C0", {}).get("hidden_fact", ""), cells=cells))
    return dict(source=src, items=out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        print(json.dumps(examples(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)))
    else:
        print(json.dumps(snapshot()))
