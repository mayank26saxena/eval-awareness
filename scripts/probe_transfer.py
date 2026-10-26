"""Do probes trained on our controlled dataset transfer to REAL eval vs deployment transcripts?

Natural data: EvalDetectBench / aware-bench (el7982/aware-bench, CC BY-NC-SA 4.0)
  eval   = 155 prompts from 15 chat-eval benchmarks (do_not_answer, xstest, civics, ...)
  deploy = real user chats (wildbench, reddit, toxic_chat, author chats, ...), agentic logs excluded
Only the first system/user turn is used, so the input matches our single-turn prompts.

For every layer we train on our v2 controlled data and test on the natural set:
  context probe   (eval cues vs none)
  format probe    (benchmark template vs natural prose)
  naive probe     (trained only where format and context agree, like most eval/deploy datasets)
and report AUROC for predicting the natural eval/deploy label. If the FORMAT probe transfers as well as
the context probe, real eval benchmarks are mostly distinguishable by format.

Reference: an in-domain probe trained on the natural data itself, with whole sources held out.

Usage:
  python scripts/probe_transfer.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from common import ROOT, read_jsonl, slug
from probe import auc, clf, extract

DEPLOY_EXCLUDE = {"openhands", "traced_run", "dataclaw"}  # long agentic logs, not comparable to single turns
MAX_CHARS = 2000


def first_turn(record):
    msgs = []
    for m in record["input"]:
        if m["role"] in ("system", "user") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"][:MAX_CHARS]})
        if m["role"] == "user":
            break
    return msgs if msgs and msgs[-1]["role"] == "user" else None


def natural_items(seed=0):
    ev = json.load(open(ROOT / "data/external/awarebench_chat_evals.json"))
    dep = [r for r in json.load(open(ROOT / "data/external/awarebench_deploy.json"))
           if r["metadata"]["category"] not in DEPLOY_EXCLUDE]
    items = []
    for label, pool in ((1, ev), (0, dep)):
        for r in pool:
            msgs = first_turn(r)
            if msgs:
                # Source datasets reuse some ids across splits, so prefix with the split to keep them unique.
                items.append({"id": f"{'eval' if label else 'deploy'}:{r['id']}", "messages": msgs, "label": label,
                              "source": r["metadata"]["category"]})
    rng = random.Random(seed)
    evals = [i for i in items if i["label"] == 1]
    deps = [i for i in items if i["label"] == 0]
    rng.shuffle(deps)
    return evals + deps[: len(evals)]  # balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default=str(ROOT / "data/rendered/v2_main.jsonl"))
    args = ap.parse_args()
    outdir = ROOT / "results" / slug(args.model)

    train = [it for it in read_jsonl(args.train) if it["behavior"] != "topic_control"]
    nat = natural_items()
    print(f"train: {len(train)} controlled items | natural: {sum(i['label'] for i in nat)} eval + "
          f"{sum(1 - i['label'] for i in nat)} deploy from {len({i['source'] for i in nat})} sources")

    train_stem = Path(args.train).stem
    A_tr = extract(args.model, train, outdir / f"activations_{train_stem}_behavior.npz")
    A_nat = extract(args.model, nat, outdir / "activations_awarebench_natural.npz")

    ctx = np.array([it["context"] == "eval" for it in train], dtype=int)
    fmt = np.array([it["format"] == "benchmark" for it in train], dtype=int)
    diag = ctx == fmt
    y = np.array([i["label"] for i in nat])
    src = np.array([i["source"] for i in nat])

    res = {"model": args.model, "n_natural": len(nat), "layers": {}}
    print("layer | trained on ours -> natural eval/deploy AUROC: context  format  naive | in-domain (held-out sources)")
    for L in range(A_tr.shape[1]):
        X, Xn = A_tr[:, L], A_nat[:, L]
        r = {
            "context_probe": auc(y, clf().fit(X, ctx).predict_proba(Xn)[:, 1]),
            "format_probe": auc(y, clf().fit(X, fmt).predict_proba(Xn)[:, 1]),
            "naive_probe": auc(y, clf().fit(X[diag], ctx[diag]).predict_proba(Xn)[:, 1]),
        }
        s = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Xn, y, src):
            s[te] = clf().fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
        r["in_domain"] = auc(y, s)
        res["layers"][L] = r
        print(f"{L:5d} | {r['context_probe']:.2f}  {r['format_probe']:.2f}  {r['naive_probe']:.2f} | {r['in_domain']:.2f}")
    (outdir / f"probe_transfer_awarebench_{train_stem}.json").write_text(json.dumps(res, indent=2))
    print(f"-> {outdir / f'probe_transfer_awarebench_{train_stem}.json'}")


if __name__ == "__main__":
    main()
