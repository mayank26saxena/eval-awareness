"""Activation steering on the residual stream, with matched control directions.

Directions are fitted on last-prompt-token activations at `hidden_states[L]` (the probe cache from
scripts/probe.py), leaving out the steered task so no direction has seen the prompt it is applied to:
  context_probe   logistic-regression context probe (eval vs deploy), mapped back to raw activation space
  format_probe    the same for format (benchmark vs natural)
  context_mean    contrastive-prompt direction: mean(eval) - mean(deploy) activations
  random_<k>      random Gaussian directions (seeded)
Each direction is unit-normalized and added as alpha * norm_unit * d to the output of decoder block L - 1
(= hidden_states[L]) at every position, prompt and generated tokens. norm_unit is the mean activation norm
at the last prompt token at that layer. Positive alpha points toward eval (context) or benchmark (format).

For a coherence control, each response's mean token negative log-likelihood under the unsteered model is
recorded (`nll`). Decoding is greedy by default; with --samples N, N responses per cell are sampled with the
model's default generation settings (seeded per prompt, direction and alpha) and numbered by `sample`.

Usage:
  python scripts/steer.py --data data/rendered/v3_main.jsonl --behavior pii --explicit absent \
      --alphas -2 -1 0 1 2 --out results/qwen2.5-7b-instruct/steer_v3_pii_L10.jsonl
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_model, prompt_text, read_jsonl, slug


def unit(v):
    return v / np.linalg.norm(v)


def probe_direction(X, y):
    sc = StandardScaler().fit(X)
    w = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(X), y).coef_[0]
    return unit(w / sc.scale_)  # gradient of the logit w.r.t. raw activations


def directions(acts, items, layer, held_out_task, n_random, d_model):
    keep = [i for i, it in enumerate(items) if it["behavior"] != "topic_control" and it["task_id"] != held_out_task]
    X = acts[keep, layer]
    ctx = np.array([items[i]["context"] == "eval" for i in keep], dtype=int)
    fmt = np.array([items[i]["format"] == "benchmark" for i in keep], dtype=int)
    out = {"context_probe": probe_direction(X, ctx), "format_probe": probe_direction(X, fmt),
           "context_mean": unit(X[ctx == 1].mean(0) - X[ctx == 0].mean(0))}
    rng = np.random.default_rng(0)  # same random directions for every task
    for k in range(n_random):
        out[f"random_{k}"] = unit(rng.standard_normal(d_model))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data/rendered/v3_main.jsonl"))
    ap.add_argument("--layer", type=int, default=10, help="hidden_states index; steering hooks block layer-1")
    ap.add_argument("--behavior", default="pii")
    ap.add_argument("--explicit", default="absent")
    ap.add_argument("--contexts", nargs="+", default=["deploy", "eval"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-2, -1, 0, 1, 2])
    ap.add_argument("--directions", nargs="+", default=["context_probe", "format_probe", "context_mean",
                                                         "random_0", "random_1", "random_2"])
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--samples", type=int, default=0, help="0 = greedy; N = N sampled responses per cell")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = read_jsonl(args.data)
    z = np.load(ROOT / "results" / slug(args.model) / f"activations_{Path(args.data).stem}.npz", allow_pickle=True)
    assert list(z["ids"]) == [it["id"] for it in items], "activation cache does not match the dataset"
    acts = z["acts"]
    norm_unit = float(np.linalg.norm(acts[:, args.layer], axis=1).mean())
    n_random = sum(d.startswith("random_") for d in args.directions)
    targets = [it for it in items if it["behavior"] == args.behavior and it["explicit"] == args.explicit
               and it["context"] in args.contexts][: args.limit]

    tok, model = load_model(args.model)
    block = model.model.layers[args.layer - 1]
    state = {"vec": None}

    def hook(_mod, _inp, out):
        if state["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + state["vec"],) + out[1:]
        return out + state["vec"]

    block.register_forward_hook(hook)
    out_path = Path(args.out)
    done = {(r["id"], r["direction"], r["alpha"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.pad_token_id}
    cos_logged = False
    t0, n = time.time(), 0
    with out_path.open("a") as f:
        for it in targets:
            dirs = directions(acts, items, args.layer, it["task_id"], n_random, acts.shape[-1])
            if not cos_logged:
                print(f"norm unit {norm_unit:.1f}; cos(context_probe, format_probe) = "
                      f"{dirs['context_probe'] @ dirs['format_probe']:+.3f}; cos(context_probe, context_mean) = "
                      f"{dirs['context_probe'] @ dirs['context_mean']:+.3f}", flush=True)
                cos_logged = True
            enc = tok(prompt_text(tok, it), return_tensors="pt", add_special_tokens=False).to(model.device)
            n_prompt = enc["input_ids"].shape[1]
            for name in args.directions:
                for a in args.alphas:
                    if a == 0 and name != args.directions[0]:
                        continue  # one unsteered baseline per prompt
                    key = (it["id"], name if a else "none", a)
                    if key in done:
                        continue
                    state["vec"] = (torch.tensor(dirs[name] * a * norm_unit, dtype=torch.bfloat16, device=model.device)
                                    if a else None)
                    sample_kw = (dict(do_sample=True, num_return_sequences=args.samples) if args.samples
                                 else dict(do_sample=False))
                    torch.manual_seed(zlib.crc32(repr(key).encode()))
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                                             **sample_kw)
                        state["vec"] = None
                        for k, seq in enumerate(gen):
                            tgt = seq[n_prompt:].tolist()
                            end = next((j for j, t in enumerate(tgt) if t in eos), len(tgt))
                            seq = seq[: n_prompt + min(end + 1, len(tgt))]  # keep end-of-turn, drop padding
                            nll = float("nan")
                            if len(seq) > n_prompt:
                                logits = model(seq[None]).logits[0, n_prompt - 1 : -1].float()
                                nll = float(torch.nn.functional.cross_entropy(logits, seq[n_prompt:]).item())
                            row = {k_: it[k_] for k_ in ("id", "task_id", "behavior", "format", "context", "explicit")}
                            row.update(model=args.model, layer=args.layer, direction=key[1], alpha=a, nll=nll,
                                       response=tok.decode(seq[n_prompt:], skip_special_tokens=True).strip())
                            if args.samples:
                                row["sample"] = k
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    print(f"  {n} done, {it['id']} {key[1]} {a:+g}  nll {nll:.2f}  {time.time() - t0:.0f}s", flush=True)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
