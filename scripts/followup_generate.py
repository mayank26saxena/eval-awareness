"""Sample replies for the follow-up experiment with vLLM (one model per call). Resumable; seeds are a pure function of
(model, item, cell, k).

  python scripts/followup_generate.py --model Qwen/Qwen2.5-7B-Instruct --data data/followup/rendered/pilot.jsonl \
      --out results/followup/pilot/qwen2.5-7b-instruct.jsonl --k 4 [--cells C0 C7 C8] [--limit N]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

TEMP, TOP_P, MAX_NEW = 0.7, 0.95, 384


def seed_of(model, item_id, k):
    return int(hashlib.sha256(f"{model}|{item_id}|{k}".encode()).hexdigest(), 16) % (2**31 - 1)


def chat_prompt(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        if "system" not in str(e).lower() or msgs[0]["role"] != "system":
            raise
        msgs = [dict(msgs[1], content=msgs[0]["content"] + "\n\n" + msgs[1]["content"])] + msgs[2:]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N items per benchmark (smoke tests only)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    if a.cells:
        rows = [r for r in rows if r["cell"] in a.cells]
    if a.limit:
        keep, seen = [], {}
        for r in rows:
            seen.setdefault(r["benchmark"], [])
            if r["task_id"] not in seen[r["benchmark"]]:
                seen[r["benchmark"]].append(r["task_id"])
            if r["task_id"] in seen[r["benchmark"]][: a.limit]:
                keep.append(r)
        rows = keep
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["id"], r["k"]) for r in map(json.loads, open(out))}
    todo = [(r, k) for r in rows for k in range(a.k) if (r["id"], k) not in done]
    print(f"{len(rows)} prompts x {a.k} samples; {len(done)} done; {len(todo)} to do -> {out}", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    t0, n_tok = time.time(), 0
    with open(out, "a") as f:
        for i in range(0, len(todo), a.chunk):
            part = todo[i:i + a.chunk]
            prompts = [chat_prompt(tok, r["messages"]) for r, _ in part]
            sps = [SamplingParams(temperature=TEMP, top_p=TOP_P, max_tokens=MAX_NEW, seed=seed_of(a.model, r["id"], k))
                   for r, k in part]
            res = llm.generate(prompts, sps, use_tqdm=False)
            for (r, k), o in zip(part, res):
                c = o.outputs[0]
                n_tok += len(c.token_ids)
                f.write(json.dumps(dict(id=r["id"], task_id=r["task_id"], cell=r["cell"], benchmark=r["benchmark"],
                                        model=a.model, k=k, response=c.text, finish=c.finish_reason,
                                        n_prompt_tokens=len(o.prompt_token_ids), n_new_tokens=len(c.token_ids)),
                                   ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"{min(i + a.chunk, len(todo))}/{len(todo)}  {el:.0f}s  {n_tok / el:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
