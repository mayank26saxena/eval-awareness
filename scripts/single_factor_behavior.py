"""Behavioral effect of each single context cue (v3 single-factor set), with the full-context effect for reference.

For every task and format, compares the response with one factor flipped against the all-realistic response,
and reports the mean paired change in the good-behavior rate per behavior, with a bootstrap 95% CI over
tasks (both formats of a task are resampled together). The full-context effect (all context cues vs none,
v3 main grid, explicit absent) is computed the same way for comparison.

Usage:
  python scripts/single_factor_behavior.py results/qwen2.5-7b-instruct/judged_v3_single.jsonl \
      --main results/qwen2.5-7b-instruct/judged_v3_main.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import read_jsonl

FACTORS = ["F1", "F2", "F4", "F6", "F8"]


def boot(per_task, n=5000, seed=0):
    """Mean over tasks of the per-task mean difference, with a bootstrap CI over tasks."""
    rng = random.Random(seed)
    vals = [sum(d) / len(d) for d in per_task.values()]
    mean = sum(vals) / len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return mean, bs[int(0.025 * n)], bs[int(0.975 * n) - 1], sum(v > 0 for v in vals), sum(v < 0 for v in vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("single")
    ap.add_argument("--main", default=None)
    args = ap.parse_args()

    rows = read_jsonl(args.single)
    good = {r["id"]: r["good"] for r in rows if r.get("good") is not None}
    beh = {r["task_id"]: r["behavior"] for r in rows}
    out = {}
    for b in sorted(set(beh.values())):
        print(f"\n{b}")
        tasks = [t for t in beh if beh[t] == b]
        base = [good.get(f"{t}__{f}__flip_none") for t in tasks for f in ("natural", "benchmark")]
        base = [x for x in base if x is not None]
        print(f"  all-realistic good rate {sum(base) / len(base):.2f} (n={len(base)})")
        out[b] = {"baseline": sum(base) / len(base)}
        for fac in FACTORS:
            per_task = defaultdict(list)
            for t in tasks:
                for f in ("natural", "benchmark"):
                    a, z = good.get(f"{t}__{f}__flip_{fac}"), good.get(f"{t}__{f}__flip_none")
                    if a is not None and z is not None:
                        per_task[t].append(int(a) - int(z))
            m, lo, hi, up, down, n = boot(per_task)
            out[b][fac] = {"effect": m, "ci": [lo, hi], "tasks_up": up, "tasks_down": down, "n_tasks": n}
            print(f"  {fac}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  tasks up {up} / down {down} of {n}")
        if args.main:
            mg = {r["id"]: r["good"] for r in read_jsonl(args.main) if r.get("good") is not None}
            per_task = defaultdict(list)
            for t in tasks:
                for f in ("natural", "benchmark"):
                    a, z = mg.get(f"{t}__{f}__eval__absent"), mg.get(f"{t}__{f}__deploy__absent")
                    if a is not None and z is not None:
                        per_task[t].append(int(a) - int(z))
            m, lo, hi, up, down, n = boot(per_task)
            out[b]["all_context_cues"] = {"effect": m, "ci": [lo, hi], "tasks_up": up, "tasks_down": down, "n_tasks": n}
            print(f"  all context cues (main grid, explicit absent): {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"tasks up {up} / down {down} of {n}")
    dest = Path(args.single).with_name(Path(args.single).stem + "_factor_effects.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
