"""Confound checks for the probe-transfer result (run after probe_transfer.py).

For a context probe trained on the controlled set and applied to natural eval/deploy transcripts, reports:
  - AUROC of prompt length alone (a trivial baseline)
  - probe AUROC on English-only prompts (non-English eval benchmarks are a possible shortcut)
  - mean AUROC within prompt-length quintiles (length-matched)
  - mean probe score per source

Usage:
  python scripts/transfer_confounds.py --model Qwen/Qwen2.5-7B-Instruct --train data/rendered/v2_main.jsonl --layers 10 14 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, read_jsonl, slug
from probe import auc, clf
from probe_transfer import natural_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default=str(ROOT / "data/rendered/v2_main.jsonl"))
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 14, 20])
    args = ap.parse_args()

    outdir = ROOT / "results" / slug(args.model)
    stem = Path(args.train).stem
    nat = natural_items()
    y = np.array([i["label"] for i in nat])
    src = np.array([i["source"] for i in nat])
    text = [i["messages"][-1]["content"] for i in nat]
    length = np.array([len(t) for t in text])
    english = np.array([sum(c.isascii() for c in t) / max(1, len(t)) > 0.95 for t in text])

    train = [it for it in read_jsonl(args.train) if it["behavior"] != "topic_control"]
    ctx = np.array([it["context"] == "eval" for it in train], dtype=int)
    A_tr = np.load(outdir / f"activations_{stem}_behavior.npz")["acts"]
    A_nat = np.load(outdir / "activations_awarebench_natural.npz")["acts"]

    res = {
        "model": args.model,
        "train": stem,
        "length_only_auroc": auc(y, length),
        "english_share": {"eval": float(english[y == 1].mean()), "deploy": float(english[y == 0].mean())},
        "layers": {},
    }
    print(f"length-only AUROC: {res['length_only_auroc']:.2f} | English share eval/deploy: "
          f"{res['english_share']['eval']:.2f} / {res['english_share']['deploy']:.2f}")
    bins = np.digitize(length, np.quantile(length, [0.2, 0.4, 0.6, 0.8]))
    for L in args.layers:
        s = clf().fit(A_tr[:, L], ctx).predict_proba(A_nat[:, L])[:, 1]
        within = [auc(y[bins == k], s[bins == k]) for k in range(5) if len(set(y[bins == k])) > 1]
        r = {
            "all": auc(y, s),
            "english_only": auc(y[english], s[english]),
            "within_length_quintiles": float(np.mean(within)),
            "corr_score_length": float(np.corrcoef(s, length)[0, 1]),
            "per_source_mean_p_eval": {k: float(s[src == k].mean()) for k in sorted(set(src))},
        }
        res["layers"][L] = r
        print(f"L{L}: all {r['all']:.2f} | English-only {r['english_only']:.2f} | "
              f"length-matched {r['within_length_quintiles']:.2f} | corr(score, length) {r['corr_score_length']:.2f}")
    out = outdir / f"transfer_confounds_{stem}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
