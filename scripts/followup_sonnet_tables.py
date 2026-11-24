"""Write Sonnet-graded copies of the judged reply files (deviation 11): 'good' = Sonnet's verdict (True = protected,
None = UNCLEAR), the automated judge's verdict kept as 'judge_good'. Every analysis script then runs unchanged on
results/followup/sonnet_full/judged/{main,steer}/ with no correction (Sonnet is the grader).
  python3 scripts/followup_sonnet_tables.py
"""
import glob, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results_pod/followup/results/followup"
S = ROOT / "results/followup/sonnet_full"
for name, pat in (("main", "main/judged_*.jsonl"), ("steer", "steer/judged_*.jsonl")):
    lab = {}
    for l in open(S / f"labels_{name}.jsonl"):
        r = json.loads(l)
        lab[(r["id"], r["model"], r["k"])] = r["claude_good"]
    out = S / "judged" / name
    out.mkdir(parents=True, exist_ok=True)
    n = u = 0
    for p in sorted(glob.glob(str(R / pat))):
        with open(out / Path(p).name, "w") as f:
            for l in open(p):
                r = json.loads(l)
                r["judge_good"] = r.get("good")
                r["good"] = lab.get((r["id"], r["model"], r["k"]))
                n += 1; u += r["good"] is None
                f.write(json.dumps(r) + "\n")
    print(f"{name}: {n} replies, {u} without a Sonnet verdict (UNCLEAR) -> {out}")
