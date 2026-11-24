"""List example candidates for the write-up: items where one single change flips the model from leaking on every sample
(judge) to protecting on every sample. Prints counts per contrast and the shortest candidates. Selection is by rule, so
the examples shown are representative of consistent flips, not hand-picked single samples."""
import glob, json, sys
from collections import defaultdict
R = "results_pod/followup/results/followup"
rows = [json.loads(l) for p in glob.glob(f"{R}/main/judged_*.jsonl") if "olmo" not in p for l in open(p)]
g = defaultdict(list)
for r in rows:
    g[(r["model"], r["task_id"], r["cell"])].append(r)
pairs = {"N placeholders": ("C0", "C1"), "P system": ("C0", "C8"), "P user turn": ("C8", "C10"), "S format": ("C0", "C2"), "E note": ("C0", "C3")}
out = {}
for name, (a, b) in pairs.items():
    c = []
    for (m, t, cell), v in g.items():
        if cell != a or len(v) < 4 or any(x["good"] for x in v):
            continue
        w = g.get((m, t, b), [])
        if len(w) == 4 and all(x["good"] for x in w):
            c.append(dict(model=m, task_id=t, bench=v[0]["benchmark"], len=sum(len(x["response"]) for x in v + w)))
    c.sort(key=lambda x: x["len"])
    out[name] = c
    print(f"{name} ({a}->{b}): {len(c)} items flip 0/4 -> 4/4;", "by bench", {k: sum(x['bench'] == k for x in c) for k in ("confaide", "privacylens")})
    for x in c[:6]:
        print("   ", x["model"].split("/")[-1], x["task_id"], x["bench"], x["len"])
st = [json.loads(l) for p in glob.glob(f"{R}/steer/judged_*.jsonl") for l in open(p)]
s = defaultdict(list)
for r in st:
    s[(r["model"], r["task_id"], r["condition"])].append(r)
c = [(m, t) for (m, t, cond), v in s.items() if cond == "base" and not any(x["good"] for x in v)
     and all(x["good"] for x in s[(m, t, "steer")]) and not any(x["good"] for x in s[(m, t, "random")])]
print(f"steer (base 0/2, random 0/2 -> steer 2/2): {len(c)}", [f"{m.split('/')[-1]}:{t}" for m, t in c[:8]])
json.dump(out, open("/tmp/claude-501/-Users-mayanksaxena-Desktop/3c39535f-266d-4e07-b671-7616306606ed/scratchpad/cands.json", "w"))
