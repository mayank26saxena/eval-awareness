"""Generate model responses for rendered dataset items.

Greedy decoding by default so runs are reproducible. Resumes if the output file already has some ids.

Usage:
  python scripts/generate.py --model google/gemma-3-4b-it --data data/rendered/pilot_main.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch

from common import ROOT, load_model, prompt_text, read_jsonl, slug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--data", default=str(ROOT / "data/rendered/pilot_main.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=350)
    ap.add_argument("--batch-size", type=int, default=1, help="1 = deterministic; larger batches change greedy outputs on MPS")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = read_jsonl(args.data)[: args.limit]
    data_name = Path(args.data).stem
    out = Path(args.out) if args.out else ROOT / "results" / slug(args.model) / f"responses_{data_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to generate -> {out}")
    if not todo:
        return

    tok, model = load_model(args.model)
    t0 = time.time()
    with open(out, "a") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            texts = [prompt_text(tok, it) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for it, ids in zip(batch, new):
                resp = tok.decode(ids, skip_special_tokens=True).strip()
                row = {k: it[k] for k in ("id", "task_id", "behavior", "format", "context", "explicit", "factors", "n_context_cues")}
                row.update(model=args.model, response=resp)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n = min(i + args.batch_size, len(todo))
            print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
