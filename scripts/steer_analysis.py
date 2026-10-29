"""Summarize a judged steering sweep (scripts/steer.py + scripts/judge.py).

For each direction and alpha: the privacy-protection rate (judge GOOD) on deployment-context and
evaluation-context prompts, its change from the unsteered baseline (paired by prompt, bootstrap CI over
tasks), and coherence: mean NLL under the unsteered model and the fraction of degenerate responses
(distinct-bigram ratio < 0.5 or empty). Changes are also reported restricted to coherent responses.

Usage:
  python scripts/steer_analysis.py results/qwen2.5-7b-instruct/steer_v3_pii_L10.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import read_jsonl


def degenerate(text):
    w = text.split()
    if len(w) < 5:
        return True
    bigrams = list(zip(w, w[1:]))
    return len(set(bigrams)) / len(bigrams) < 0.5


def boot(per_task, n=5000, seed=0):
    vals = [sum(v) / len(v) for v in per_task.values() if v]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged")
    args = ap.parse_args()
    rows = [r for r in read_jsonl(args.judged) if r.get("good") is not None]
    for r in rows:
        r["degenerate"] = degenerate(r["response"])
    base = {r["id"]: r for r in rows if r["direction"] == "none"}
    out = {}
    for ctx in ("deploy", "eval"):
        b = [base[i]["good"] for i in base if base[i]["context"] == ctx]
        print(f"\n{ctx} prompts: unsteered protection {sum(b) / len(b):.2f} (n={len(b)})")
        print(f"  {'direction':14s} {'alpha':>6s}  protect  change [95% CI]          coherent-only change   nll   degenerate")
        out[ctx] = {"baseline": sum(b) / len(b), "cells": []}
        for d in sorted({r["direction"] for r in rows} - {"none"}):
            for a in sorted({r["alpha"] for r in rows if r["direction"] == d}):
                cell = [r for r in rows if r["direction"] == d and r["alpha"] == a and r["context"] == ctx]
                diff, diff_coh = defaultdict(list), defaultdict(list)
                for r in cell:
                    delta = int(r["good"]) - int(base[r["id"]]["good"])
                    diff[r["task_id"]].append(delta)
                    if not r["degenerate"]:
                        diff_coh[r["task_id"]].append(delta)
                m, lo, hi = boot(diff)
                mc, _, _ = boot(diff_coh)
                rate = sum(r["good"] for r in cell) / len(cell)
                nll = sum(r["nll"] for r in cell) / len(cell)
                deg = sum(r["degenerate"] for r in cell) / len(cell)
                out[ctx]["cells"].append({"direction": d, "alpha": a, "protect": rate, "change": m, "ci": [lo, hi],
                                          "change_coherent": mc, "nll": nll, "degenerate": deg, "n": len(cell)})
                print(f"  {d:14s} {a:+6.2f}  {rate:6.2f}  {m:+.2f} [{lo:+.2f}, {hi:+.2f}]          {mc:+.2f}"
                      f"          {nll:5.2f}  {deg:5.2f}")
    # Sign-specific effect: protection(+alpha) - protection(-alpha), paired by prompt, all prompts pooled.
    # A generic perturbation effect cancels; a context effect should be positive (+alpha points toward eval).
    print(f"\nsign-specific effect, protect(+a) - protect(-a), all 40 prompts")
    out["antisymmetric"] = []
    for d in sorted({r["direction"] for r in rows} - {"none"}):
        for a in sorted({r["alpha"] for r in rows if r["direction"] == d and r["alpha"] > 0}):
            g = {(r["id"], r["alpha"]): r for r in rows if r["direction"] == d}
            per_task = defaultdict(list)
            for i in base:
                if (i, a) in g and (i, -a) in g:
                    per_task[base[i]["task_id"]].append(int(g[i, a]["good"]) - int(g[i, -a]["good"]))
            m, lo, hi = boot(per_task)
            out["antisymmetric"].append({"direction": d, "alpha": a, "effect": m, "ci": [lo, hi]})
            print(f"  {d:14s} {a:4.2f}  {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    dest = Path(args.judged).with_suffix(".summary.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
