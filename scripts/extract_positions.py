"""Extract activations at multiple token positions of full transcripts (prompt + model response).

For each item, the transcript is the chat-formatted prompt followed by the model's own response
(from scripts/generate.py), or the prompt alone when --responses is omitted. One forward pass yields
hidden states at:
  system_end     last token of the system message content
  user_end       last token of the user message content
  pre_response   last prompt token (where the response begins)
  response       N response tokens sampled uniformly (seeded), excluding the end-of-turn token

Output: an .npz with `acts` [m, d] at one layer and `ids` "<item_id>|<position_type>|<token_index>",
plus a .jsonl with per-position metadata. The output feeds scripts/nla_explain.py directly.

Usage:
  python scripts/extract_positions.py --model Qwen/Qwen2.5-7B-Instruct --data data/rendered/v3_main.jsonl \
      --responses results/qwen2.5-7b-instruct/responses_v3_main.jsonl --layer 21 --n-response 8
  python scripts/extract_positions.py --data data/rendered/v3_single.jsonl   # prompt positions only
"""

import argparse
import json
import random

import numpy as np
import torch

from common import ROOT, load_model, prompt_text, read_jsonl, slug


def content_end_token(offsets, text, content, search_from=0):
    """Index of the token whose span ends at the end of `content` within `text`."""
    start = text.find(content, search_from)
    assert start >= 0, "message content not found in chat-formatted text"
    end = start + len(content)
    cands = [i for i, (a, b) in enumerate(offsets) if a < end <= b]
    return cands[0], end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--responses", default=None, help="omit to read prompt positions only")
    ap.add_argument("--layer", type=int, default=21, help="hidden_states index (NLA layer K -> K + 1)")
    ap.add_argument("--n-response", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = {it["id"]: it for it in read_jsonl(args.data)}
    responses = {r["id"]: r["response"] for r in read_jsonl(args.responses)} if args.responses else None
    ids = [i for i in items if responses is None or i in responses]
    tok, model = load_model(args.model)
    rng = random.Random(args.seed)
    stem = f"positions_{slug(args.data.split('/')[-1].replace('.jsonl', ''))}_L{args.layer}"
    out = args.out or str(ROOT / "results" / slug(args.model) / stem)

    acts, keys, meta = [], [], []
    for n, iid in enumerate(ids):
        it = items[iid]
        ptext = prompt_text(tok, it)
        full = ptext + responses[iid] + tok.eos_token if responses else ptext
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        n_prompt = len(tok(ptext, add_special_tokens=False)["input_ids"])

        positions = []
        cursor = 0
        for m in it["messages"]:
            if m["role"] in ("system", "user"):
                idx, cursor = content_end_token(offsets, full, m["content"], cursor)
                positions.append((f"{m['role']}_end", idx))
        positions.append(("pre_response", n_prompt - 1))
        resp = list(range(n_prompt, len(enc["input_ids"]) - 1)) if responses else []  # drop the end-of-turn token
        for idx in sorted(rng.sample(resp, min(args.n_response, len(resp)))):
            positions.append(("response", idx))

        with torch.no_grad():
            hs = model(torch.tensor([enc["input_ids"]], device=model.device),
                       output_hidden_states=True).hidden_states[args.layer][0]
        for ptype, idx in positions:
            acts.append(hs[idx].float().cpu().numpy())
            keys.append(f"{iid}|{ptype}|{idx}")
            meta.append({"key": keys[-1], "id": iid, "position_type": ptype, "token_index": idx,
                         "token": tok.decode([enc["input_ids"][idx]]), "n_prompt_tokens": n_prompt,
                         "n_total_tokens": len(enc["input_ids"])})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(ids)} items, {len(keys)} positions", flush=True)

    np.savez_compressed(out + ".npz", acts=np.stack(acts), ids=np.array(keys))
    with open(out + ".jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(keys)} positions from {len(ids)} items -> {out}.npz")


if __name__ == "__main__":
    main()
