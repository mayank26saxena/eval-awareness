"""Pod-side disk guard: the /workspace volume is 150 GB and the five generators plus the 32B judge need about 150 GB of
weights. After a model's main-run generation is reported done, delete its cached weights (they can be re-downloaded
for later stages). Never touches the judge. Loops until the driver and queue have exited."""
import re
import subprocess
import time

from huggingface_hub import scan_cache_dir

SLUG = {"qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct", "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3", "gemma-2-9b-it": "google/gemma-2-9b-it",
        "olmo-2-1124-7b-instruct": "allenai/OLMo-2-1124-7B-Instruct"}
STATUS = "/workspace/ea/logs/followup/status"
freed = set()
REV = "/workspace/ea/results/followup/env/model_revisions.jsonl"


def record_revisions():
    """Append every cached repo's resolved commit hash (first time seen), so each run is pinned to exact weights."""
    import json
    import os
    os.makedirs(os.path.dirname(REV), exist_ok=True)
    seen = set()
    if os.path.exists(REV):
        seen = {(json.loads(l)["repo_id"], json.loads(l)["revision"]) for l in open(REV)}
    with open(REV, "a") as f:
        for repo in scan_cache_dir().repos:
            for r in repo.revisions:
                if (repo.repo_id, r.commit_hash) not in seen:
                    f.write(json.dumps(dict(repo_id=repo.repo_id, revision=r.commit_hash, refs=sorted(r.refs),
                                            seen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")


def say(msg):
    with open(STATUS, "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " diskguard: " + msg + "\n")


while True:
    record_revisions()
    text = open(STATUS).read()
    done = {SLUG[s] for s in re.findall(r"main ([\w.\-]+) generated", text) if s in SLUG}
    busy = subprocess.run("ps -eo args | grep -E 'followup_(generate|judge)' | grep -v grep", shell=True,
                          capture_output=True, text=True).stdout
    for repo_id in sorted(done - freed):
        if repo_id in busy:
            continue
        info = scan_cache_dir()
        revs = [r.commit_hash for repo in info.repos if repo.repo_id == repo_id for r in repo.revisions]
        if revs:
            strat = info.delete_revisions(*revs)
            strat.execute()
            say(f"freed {strat.expected_freed_size_str} ({repo_id})")
        freed.add(repo_id)
    alive = subprocess.run("pgrep -f 'followup_(driver|queue)\\.sh'", shell=True, capture_output=True).returncode == 0
    if not alive:
        break
    time.sleep(120)
