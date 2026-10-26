# Running on a cloud GPU (27B+ models)

The Mac (M4 Pro, 24 GB) handles models up to about 8B. The behavior question needs a model that actually shows eval awareness (≥ ~27B), and the headline model, **Gemma-3-27B-IT**, has a released NLA at layer 41.

## What you need (your steps)
1. **A GPU box:** 1× H100 80GB or A100 80GB on RunPod, Lambda or Vast. That's about $2–3 per hour, and a full pilot run takes 2–3 GPU-hours.
2. **Hugging Face access for Gemma** (gated):
   - Accept the license at https://huggingface.co/google/gemma-3-27b-it (the same click also covers 4B and 12B).
   - Create a read token at https://huggingface.co/settings/tokens.
   - On the box, run `hf auth login` and paste the token. Never put it in code or commit it.
3. **Check Amazon policy** on personal cloud spend and third-party compute for side research.

## Setup on the box
```bash
git clone <your copy of this folder>   # or scp -r ~/Desktop/projects/eval-awareness user@box:
cd eval-awareness
pip install uv && uv venv --python 3.12 && uv pip install -r pyproject.toml
hf auth login
```
On CUDA, `scripts/common.py` picks `cuda` automatically. For faster generation, larger batches are fine on CUDA, but keep batch size 1 if you want results exactly comparable to the Mac runs.

## Runs (same scripts as on the Mac)
```bash
M=google/gemma-3-27b-it
python scripts/generate.py --model $M --data data/rendered/v2_main.jsonl
python scripts/generate.py --model $M --data data/rendered/v2_single.jsonl
python scripts/judge.py results/gemma-3-27b-it/responses_v2_main.jsonl --data data/rendered/v2_main.jsonl
python scripts/probe.py --model $M --data data/rendered/v2_main.jsonl
python scripts/probe_transfer.py --model $M
```
Then copy `results/gemma-3-27b-it/` back to the Mac. Activation `.npz` files for 27B are about 1–2 GB.

## Memory guide (bf16 weights)

| Model | Weights | Fits on |
|---|---|---|
| Qwen2.5-7B / Qwen3-4B | 15 / 8 GB | Mac |
| Gemma-3-12B | 24 GB | Cloud (the Mac is too tight) |
| Gemma-3-27B | 54 GB | 1× 80GB GPU |
| Llama-3.3-70B | 140 GB | 2× 80GB GPUs, or NDIF remote |

## NLA step (later)
The NLA verbalizer is a second full-size model. On one 80GB GPU:
1. Cache the target model's activations at the NLA layer (`probe.py` already saves every layer).
2. Unload the model.
3. Serve the AV with SGLang, following `kitft/natural_language_autoencoders` (`docs/inference.md`).
