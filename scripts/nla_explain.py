"""Generate Natural Language Autoencoder (NLA) explanations for cached activations.

Implements the published AV inference recipe (kitft/natural_language_autoencoders, docs/inference.md)
with Hugging Face transformers instead of SGLang, so it runs on Apple Silicon (MPS) or a single GPU:
  1. Build the AV prompt from the checkpoint's nla_meta.yaml template (chat template, tokenize=True).
  2. Locate the injection token, verifying both neighbor token ids.
  3. Rescale the activation to the sidecar's injection_scale and overwrite that token's embedding.
  4. Sample at temperature 1.0 and extract the <explanation> block.

Layer convention: the NLA's extraction_layer_index K is the OUTPUT of decoder block K, which is
Hugging Face `hidden_states[K + 1]` (index 0 is the embeddings). Our activation caches store
hidden_states, so we read index K + 1.

Usage:
  python scripts/nla_explain.py --acts results/qwen2.5-7b-instruct/activations_v3_main.npz \
      --data data/rendered/v3_main.jsonl --out results/qwen2.5-7b-instruct/nla_v3_main.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import device, read_jsonl

AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"


def load_av(repo):
    meta = yaml.safe_load(open(hf_hub_download(repo, "nla_meta.yaml")))
    tok = AutoTokenizer.from_pretrained(repo)
    tk = meta["tokens"]
    assert tok.convert_tokens_to_ids(tk["injection_char"]) == tk["injection_token_id"] or \
        tok.encode(tk["injection_char"], add_special_tokens=False) == [tk["injection_token_id"]], \
        "injection char does not map to the sidecar's token id"
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device()).eval()
    content = meta["prompt_templates"]["av"].format(injection_char=tk["injection_char"])
    ids = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                  add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    pos = [p for p, t in enumerate(ids) if t == tk["injection_token_id"]
           and ids[p - 1] == tk["injection_left_neighbor_id"] and ids[p + 1] == tk["injection_right_neighbor_id"]]
    assert len(pos) == 1, f"expected exactly one valid injection position, found {pos}"
    return meta, tok, model, torch.tensor([ids], device=model.device), pos[0]


def explain_batch(meta, tok, model, input_ids, pos, vecs, max_new_tokens, seed):
    """Explain a batch of vectors. All AV prompts are identical apart from the injected vector,
    so the batch needs no padding."""
    b = len(vecs)
    ids = input_ids.repeat(b, 1)
    emb = model.get_input_embeddings()(ids).clone()
    v = torch.tensor(np.stack(vecs), dtype=torch.float32)
    v = v * (meta["extraction"]["injection_scale"] / v.norm(dim=1, keepdim=True))
    emb[:, pos] = v.to(emb.dtype).to(emb.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                             pad_token_id=tok.eos_token_id)
    results = []
    for row in out:
        raw = tok.decode(row, skip_special_tokens=True)
        m = re.search(r"<explanation>\s*(.*?)\s*(</explanation>|$)", raw, re.DOTALL)
        results.append(((m.group(1) if m else raw).strip(), raw))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True,
                    help="npz with ids and acts: [n, layers+1, d] (layer selected from the sidecar) or "
                         "[n, d] already at the NLA layer (from extract_positions.py; ids 'item|position|token')")
    ap.add_argument("--data", required=True, help="jsonl with item metadata, aligned to the cache ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--av", default=AV_REPO)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts share one length, so batching needs no padding")
    args = ap.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    acts, ids = z["acts"], list(z["ids"])
    items = {it["id"]: it for it in read_jsonl(args.data)}
    meta, tok, model, input_ids, pos = load_av(args.av)
    layer = meta["extraction_layer_index"] + 1  # hidden_states index
    print(f"AV {args.av}: extraction layer {meta['extraction_layer_index']} -> hidden_states[{layer}], "
          f"injection at token {pos}, scale {meta['extraction']['injection_scale']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["sample"]) for r in read_jsonl(out)} if out.exists() else set()
    t0, n = time.time(), 0
    todo = [(i, iid, s) for i, iid in enumerate(ids[: args.limit]) for s in range(args.samples)
            if (iid, s) not in done]
    with out.open("a") as f:
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0 : b0 + args.batch_size]
            vecs = [acts[i] if acts.ndim == 2 else acts[i, layer] for i, _, _ in batch]
            results = explain_batch(meta, tok, model, input_ids, pos, vecs, args.max_new_tokens, seed=b0)
            for (i, iid, s), (expl, raw) in zip(batch, results):
                item_id, ptype, tidx = (iid.split("|") + [None, None])[:3]
                it = items.get(item_id, {})
                row = {"id": iid, "item_id": item_id, "position_type": ptype or "pre_response",
                       "token_index": int(tidx) if tidx else None, "sample": s,
                       "layer_index": meta["extraction_layer_index"], "explanation": expl, "raw": raw,
                       **{k: it.get(k) for k in ("task_id", "behavior", "format", "context", "explicit")}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            print(f"  {min(b0 + args.batch_size, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {n} explanations -> {out}")


if __name__ == "__main__":
    main()
