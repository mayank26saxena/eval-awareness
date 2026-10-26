"""Extract last-prompt-token activations at every layer and train linear probes.

Probes (logistic regression, cross-validated with whole tasks held out):
  context   eval vs deploy cues
  format    benchmark vs natural
  explicit  grading language present vs absent

Plus the experiment that mirrors prior work's confound:
  naive     trained only on the "diagonal" (benchmark+eval vs natural+deploy), where format and context agree,
            as most existing eval-vs-deploy datasets are. Tested on the off-diagonal cells, where they disagree.
            If it calls natural+eval "deploy" and benchmark+deploy "eval", it learned format, not context.

Usage:
  python scripts/probe.py --model google/gemma-3-4b-it
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_model, prompt_text, read_jsonl, slug


def extract(model_name, items, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["ids"]) == [it["id"] for it in items]:
            print(f"using cached activations {cache}")
            return z["acts"]
    tok, model = load_model(model_name)
    acts = []
    for i, it in enumerate(items):
        enc = tok(prompt_text(tok, it), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        acts.append(torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy())
        if (i + 1) % 20 == 0:
            print(f"  activations {i + 1}/{len(items)}", flush=True)
    acts = np.stack(acts)  # [n_items, n_layers + 1, d_model]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, acts=acts, ids=np.array([it["id"] for it in items]))
    return acts


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))


def cv_scores(X, y, groups):
    """Held-out-task probe scores for every item."""
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=len(set(groups))).split(X, y, groups):
        m = clf().fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--data", default=str(ROOT / "data/rendered/pilot_main.jsonl"))
    args = ap.parse_args()

    items = read_jsonl(args.data)
    outdir = ROOT / "results" / slug(args.model)
    stem = Path(args.data).stem
    acts = extract(args.model, items, outdir / f"activations_{stem}.npz")
    n_layers = acts.shape[1]

    beh = np.array([it["behavior"] != "topic_control" for it in items])
    ctx = np.array([it["context"] == "eval" for it in items], dtype=int)
    fmt = np.array([it["format"] == "benchmark" for it in items], dtype=int)
    exp = np.array([it["explicit"] == "present" for it in items], dtype=int)
    groups = np.array([it["task_id"] for it in items])

    res = {"model": args.model, "n_items": int(beh.sum()), "layers": {}}
    for L in range(n_layers):
        X = acts[beh, L]
        g = groups[beh]
        r = {}
        s_ctx = cv_scores(X, ctx[beh], g)
        s_fmt = cv_scores(X, fmt[beh], g)
        s_exp = cv_scores(X, exp[beh], g)
        r["context_auc"] = auc(ctx[beh], s_ctx)
        r["format_auc"] = auc(fmt[beh], s_fmt)
        r["explicit_auc"] = auc(exp[beh], s_exp)
        r["context_probe_on_format"] = auc(fmt[beh], s_ctx)  # want ~0.5
        r["format_probe_on_context"] = auc(ctx[beh], s_fmt)  # want ~0.5

        # Naive probe: train on diagonal cells (format and context agree), test on off-diagonal.
        diag = (ctx[beh] == fmt[beh])
        off = ~diag
        s_naive = np.zeros(beh.sum())
        for tr, te in GroupKFold(n_splits=len(set(g))).split(X, ctx[beh], g):
            tr_d = tr[diag[tr]]
            m = clf().fit(X[tr_d], ctx[beh][tr_d])
            s_naive[te] = m.predict_proba(X[te])[:, 1]
        r["naive_diag_auc"] = auc(ctx[beh][diag], s_naive[diag])
        y_off = ctx[beh][off]
        r["naive_offdiag_auc_vs_context"] = auc(y_off, s_naive[off])          # >0.5: follows context
        r["naive_offdiag_auc_vs_format"] = auc(fmt[beh][off], s_naive[off])   # >0.5: follows format

        # Topic controls: score with a context probe trained on all behavior items.
        topic = ~beh
        if topic.any():
            m = clf().fit(X, ctx[beh])
            r["topic_control_mean_p_eval"] = float(m.predict_proba(acts[topic, L])[:, 1].mean())
            r["deploy_mean_p_eval"] = float(m.predict_proba(X[ctx[beh] == 0])[:, 1].mean())
        res["layers"][L] = r

    best = max(res["layers"], key=lambda L: res["layers"][L]["context_auc"])
    res["best_context_layer"] = best
    (outdir / f"probe_results_{stem}.json").write_text(json.dumps(res, indent=2))

    print(f"\n{args.model}: {n_layers - 1} layers, {beh.sum()} behavior items, {(~beh).sum()} topic controls")
    print("layer  ctx   fmt   expl | ctx->fmt fmt->ctx | naive: diag offdiag-vs-ctx offdiag-vs-fmt | topic p(eval)")
    for L in range(n_layers):
        r = res["layers"][L]
        print(f"{L:5d}  {r['context_auc']:.2f}  {r['format_auc']:.2f}  {r['explicit_auc']:.2f} |"
              f"   {r['context_probe_on_format']:.2f}     {r['format_probe_on_context']:.2f}  |"
              f"        {r['naive_diag_auc']:.2f}      {r['naive_offdiag_auc_vs_context']:.2f}          {r['naive_offdiag_auc_vs_format']:.2f}     |"
              f"   {r.get('topic_control_mean_p_eval', float('nan')):.2f}")
    print(f"\nbest context layer: {best}  -> {outdir / f'probe_results_{stem}.json'}")


if __name__ == "__main__":
    main()
