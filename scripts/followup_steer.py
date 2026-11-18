"""Steering arm of the follow-up experiment (pre-registration: Qwen2.5-7B and Mistral-7B, x4, k = 2).

Six conditions on the 240 main items, all generated in this one engine (Hugging Face generate with hooks), so the
steering contrasts compare like with like:
  base       published prompt (cell C0)
  P          privacy instruction in the system prompt (cell C8)
  steer      C0 + alpha x mean shift on the top-k neurons (the aligned neuron set from the earlier study)
  steer_P    C8 + the same steering
  random     C0 + alpha x the same shift values on k random neurons (a different random set per item, as before)
  random_P   C8 + random steering
Neurons: every pair with a positive score gap is used (the follow-up items are not among the neuron pairs). The shift
is added to the input of each layer's down_proj at every position from --start on (after the chat header) and at every
generated token, exactly as in scripts/steer_neurons.py, here batched with left padding (per-row start positions).
Sampling matches the main run: temperature 0.7, top-p 0.95, 384 new tokens.

  python scripts/followup_steer.py --neurons results/qwen2.5-7b-instruct/neurons_external_aligned.json --start 5 \
      --data data/followup/rendered/main.jsonl --out results/followup/steer/qwen2.5-7b-instruct.jsonl
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from common import load_model, prompt_text

CONDS = {"base": ("C0", 0, False), "P": ("C8", 0, False), "steer": ("C0", 1, False), "steer_P": ("C8", 1, False),
         "random": ("C0", 1, True), "random_P": ("C8", 1, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--k-neurons", type=int, default=5000)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16, help="prompts per batch (x samples rows)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    res = json.loads(Path(a.neurons).read_text())
    npz = np.load(Path(a.neurons).with_suffix(".npz"))
    shifts, grads, names = torch.tensor(npz["pair_shifts"]), torch.tensor(npz["pair_grads"]), list(npz["pairs"])
    recs = {r["pair"]: r for r in res["pairs"]}
    spans = np.array([recs[n]["score_cue"] - recs[n]["score_deploy"] for n in names])
    use = [i for i in range(len(names)) if spans[i] > 0]
    n_layers, d_ff = shifts.shape[1:]
    shift = shifts[use].mean(0)
    attr = (shift * grads[use] / torch.tensor(spans[use], dtype=torch.float32)[:, None, None]).sum(0)
    top = torch.argsort(attr.flatten(), descending=True)[: a.k_neurons]
    flat = shift.flatten()
    steer_delta = torch.zeros_like(flat).index_copy_(0, top, flat[top]).view(n_layers, d_ff)
    print(f"{res['model']}: {len(use)} pairs with a positive gap; top {a.k_neurons} neurons; "
          f"shift norm on them {flat[top].norm():.3f}", flush=True)

    def random_delta(task_id):
        idx = torch.tensor(np.random.default_rng(zlib.crc32(task_id.encode())).choice(len(flat), a.k_neurons, replace=False))
        return torch.zeros_like(flat).index_copy_(0, idx, flat[idx]).view(n_layers, d_ff)

    rows = [json.loads(l) for l in open(a.data)]
    by = {(r["task_id"], r["cell"]): r for r in rows if r["cell"] in ("C0", "C8")}
    tasks = sorted({t for t, _ in by})
    if a.limit:
        tasks = [t for t in tasks if t.startswith("confaide")][: a.limit] + [t for t in tasks if t.startswith("privacy")][: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["k"]) for r in map(json.loads, open(out))} if out.exists() else set()
    jobs = [(t, c) for c in CONDS for t in tasks if any((f"{t}__steer_{c}", k) not in done for k in range(a.samples))]
    print(f"{len(tasks)} items x {len(CONDS)} conditions x {a.samples} samples; {len(jobs)} prompt-conditions to do", flush=True)
    if not jobs:
        return

    tok, model = load_model(res["model"])
    tok.padding_side = "left"
    dev = model.device
    state = dict(delta=None, first=None)

    def make_hook(layer):
        def hook(_mod, inp):
            if state["delta"] is None:
                return None
            x = inp[0]
            d = state["delta"][:, layer].to(x.dtype)[:, None, :]  # [B, 1, d_ff]
            if x.shape[1] == 1:
                return (x + d,)
            pos = torch.arange(x.shape[1], device=x.device)[None, :]
            mask = (pos >= state["first"][:, None]).to(x.dtype)[..., None]  # [B, T, 1]
            return (x + mask * d,)
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))

    steer_dev = steer_delta.to(dev)
    t0, n_done = time.time(), 0
    with open(out, "a") as f:
        for cond, (cell, scale, rnd) in CONDS.items():
            cj = [t for t, c in jobs if c == cond]
            cj.sort(key=lambda t: len(by[(t, cell)]["messages"][1]["content"]))
            for b in range(0, len(cj), a.batch):
                part = cj[b:b + a.batch]
                texts = [prompt_text(tok, by[(t, cell)]) for t in part for _ in range(a.samples)]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                pad = (enc["attention_mask"] == 0).sum(1)
                state["first"] = pad + a.start
                if scale:
                    if rnd:
                        d = torch.stack([random_delta(t) for t in part for _ in range(a.samples)]).to(dev)
                    else:
                        d = steer_dev[None].expand(len(texts), -1, -1)
                    state["delta"] = a.alpha * d
                else:
                    state["delta"] = None
                torch.manual_seed(zlib.crc32(f"{res['model']}|{cond}|{part[0]}".encode()))
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=0.7, top_p=0.95,
                                         pad_token_id=tok.pad_token_id)
                state["delta"] = None
                for j, ids in enumerate(gen[:, enc["input_ids"].shape[1]:]):
                    t, k = part[j // a.samples], j % a.samples
                    text = tok.decode(ids, skip_special_tokens=True).strip()
                    hit_eos = bool((ids == tok.eos_token_id).any()) if tok.eos_token_id is not None else None
                    f.write(json.dumps(dict(id=f"{t}__steer_{cond}", task_id=t, cell=cell, condition=cond,
                                            benchmark=by[(t, cell)]["benchmark"], model=res["model"], k=k,
                                            alpha=a.alpha * scale, random=rnd, response=text,
                                            finish="stop" if hit_eos else "length"), ensure_ascii=False) + "\n")
                f.flush()
                n_done += len(part)
                el = time.time() - t0
                print(f"{cond}: {b + len(part)}/{len(cj)}  total {n_done}/{len(jobs)}  {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
