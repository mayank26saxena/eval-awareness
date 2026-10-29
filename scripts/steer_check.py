"""Manipulation check for scripts/steer.py: does steering at layer L move the downstream context readout?

For each steered prompt, direction and alpha, adds the steering vector at hidden_states[L] (as in steer.py) and
reads the last-prompt-token activation at later layers. A context probe fitted at each readout layer (leaving
out the prompt's task) gives the logit for "eval". The shift from the unsteered logit is compared with the
natural gap between eval and deploy prompts of the same task and format.

Usage:
  python scripts/steer_check.py --alphas -1 -0.5 0.5 1 --readout 14 20 27
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_model, prompt_text, read_jsonl, slug
from steer import directions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data/rendered/v3_main.jsonl"))
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--readout", nargs="+", type=int, default=[14, 20, 27])
    ap.add_argument("--behavior", default="pii")
    ap.add_argument("--explicit", default="absent")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-1, -0.5, 0.5, 1])
    ap.add_argument("--directions", nargs="+", default=["context_probe", "format_probe", "context_mean",
                                                         "random_0", "random_1", "random_2"])
    args = ap.parse_args()

    items = read_jsonl(args.data)
    z = np.load(ROOT / "results" / slug(args.model) / f"activations_{Path(args.data).stem}.npz", allow_pickle=True)
    acts = z["acts"]
    norm_unit = float(np.linalg.norm(acts[:, args.layer], axis=1).mean())
    n_random = sum(d.startswith("random_") for d in args.directions)
    targets = [it for it in items if it["behavior"] == args.behavior and it["explicit"] == args.explicit]

    probes = {}  # (task, layer) -> fitted readout probe, leaving out the task
    for t in sorted({it["task_id"] for it in targets}):
        keep = [i for i, it in enumerate(items) if it["behavior"] != "topic_control" and it["task_id"] != t]
        y = np.array([items[i]["context"] == "eval" for i in keep], dtype=int)
        for L in args.readout:
            probes[t, L] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(acts[keep, L], y)

    tok, model = load_model(args.model)
    state = {"vec": None}

    def hook(_mod, _inp, out):
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"],) + out[1:]
        return out + state["vec"]

    model.model.layers[args.layer - 1].register_forward_hook(hook)

    def readout(it):
        enc = tok(prompt_text(tok, it), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        return {L: float(probes[it["task_id"], L].decision_function(hs[L][0, -1].float().cpu().numpy()[None])[0])
                for L in args.readout}

    base = {it["id"]: readout(it) for it in targets}
    gap = defaultdict(list)  # natural eval - deploy logit gap, per layer
    for it in targets:
        if it["context"] == "eval":
            dep = it["id"].replace("__eval__", "__deploy__")
            for L in args.readout:
                gap[L].append(base[it["id"]][L] - base[dep][L])
    shifts = defaultdict(list)
    for it in targets:
        dirs = directions(acts, items, args.layer, it["task_id"], n_random, acts.shape[-1])
        for name in args.directions:
            for a in args.alphas:
                state["vec"] = torch.tensor(dirs[name] * a * norm_unit, dtype=torch.bfloat16, device=model.device)
                r = readout(it)
                state["vec"] = None
                for L in args.readout:
                    shifts[name, a, L].append(r[L] - base[it["id"]][L])

    out = {"natural_gap": {L: float(np.mean(gap[L])) for L in args.readout}, "shifts": []}
    print("natural eval - deploy logit gap: " + "  ".join(f"L{L} {out['natural_gap'][L]:+.2f}" for L in args.readout))
    print(f"{'direction':14s} {'alpha':>6s}  " + "  ".join(f"{'L' + str(L):>12s}" for L in args.readout))
    for name in args.directions:
        for a in args.alphas:
            vals = {L: float(np.mean(shifts[name, a, L])) for L in args.readout}
            out["shifts"].append({"direction": name, "alpha": a, **{f"L{L}": v for L, v in vals.items()}})
            print(f"{name:14s} {a:+6.2f}  " + "  ".join(f"{vals[L]:+12.2f}" for L in args.readout))
    dest = ROOT / "results" / slug(args.model) / f"steer_check_v3_{args.behavior}_L{args.layer}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
