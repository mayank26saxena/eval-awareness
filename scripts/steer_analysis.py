"""Summarize a judged steering sweep (scripts/steer.py + scripts/judge.py).

For each direction and alpha: the privacy-protection rate (judge GOOD) on deployment-context and
evaluation-context prompts, its change from the unsteered baseline (paired by prompt, bootstrap CI over
tasks), and coherence: mean NLL under the unsteered model and the fraction of degenerate responses
(distinct-bigram ratio < 0.5 or empty). Changes are also reported restricted to coherent responses.
Sampled sweeps (rows with `sample`) are averaged per prompt first, so each prompt contributes its protection
probability. With five or more random directions, the sign-specific effect of each non-random direction is
placed within the distribution over random directions.

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
    cells = defaultdict(list)  # (id, direction, alpha) -> rows (one when greedy, several when sampled)
    for r in read_jsonl(args.judged):
        if r.get("good") is not None:
            cells[r["id"], r["direction"], r["alpha"]].append(r)
    rows = []
    for rs in cells.values():
        deg = [degenerate(r["response"]) for r in rs]
        coh = [r["good"] for r, d in zip(rs, deg) if not d]
        rows.append({**rs[0], "good": sum(r["good"] for r in rs) / len(rs), "degenerate": sum(deg) / len(deg),
                     "good_coherent": sum(coh) / len(coh) if coh else None,
                     "nll": sum(r["nll"] for r in rs) / len(rs)})
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
                    diff[r["task_id"]].append(r["good"] - base[r["id"]]["good"])
                    if r["good_coherent"] is not None:
                        diff_coh[r["task_id"]].append(r["good_coherent"] - base[r["id"]]["good"])
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
                    per_task[base[i]["task_id"]].append(g[i, a]["good"] - g[i, -a]["good"])
            m, lo, hi = boot(per_task)
            out["antisymmetric"].append({"direction": d, "alpha": a, "effect": m, "ci": [lo, hi]})
            print(f"  {d:14s} {a:4.2f}  {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    rand = [c for c in out["antisymmetric"] if c["direction"].startswith("random_")]
    for a in sorted({c["alpha"] for c in rand}):
        null = sorted(c["effect"] for c in rand if c["alpha"] == a)
        if len(null) < 5:
            continue
        sd = (sum((x - sum(null) / len(null)) ** 2 for x in null) / (len(null) - 1)) ** 0.5
        print(f"\nrandom null at alpha {a:.2f} ({len(null)} directions): mean {sum(null) / len(null):+.2f}, "
              f"sd {sd:.2f}, range [{null[0]:+.2f}, {null[-1]:+.2f}]")
        out.setdefault("random_null", {})[str(a)] = {"effects": null, "sd": sd}
        for c in out["antisymmetric"]:
            if c["alpha"] == a and not c["direction"].startswith("random_"):
                frac = sum(x >= c["effect"] for x in null) / len(null)
                c["frac_random_at_least"] = frac
                print(f"  {c['direction']:14s} {c['effect']:+.2f}; fraction of random directions >= this: {frac:.2f}")
    dest = Path(args.judged).with_suffix(".summary.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
