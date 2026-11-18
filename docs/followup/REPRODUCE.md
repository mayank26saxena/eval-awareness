# Reproducing the follow-up experiment

The question: which single change to a privacy-benchmark prompt changes a model's privacy behavior? The design, the
decision rules and every deviation are in `notes/followup_preregistration.md`, which also lists SHA-256 checksums of
every input, frozen in stages before any output existed (git history shows the order: `3b8582f` → `a74a87b` → `7f8424c`).

## Environment

| Component | Version |
|---|---|
| Models (generation) | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`, `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`, `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71`, `google/gemma-2-9b-it` @ `11c9b309abf73637e4b6f9a3fa1e92e615547819`, `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` |
| Model (automated judge) | `Qwen/Qwen2.5-32B-Instruct` @ `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, fp8 weights quantized at load by vLLM |
| GPU stack | one NVIDIA A40 (48 GB), driver CUDA 13.2; Python 3.12.3; vLLM 0.34.0; torch 2.14.0 (cu130); transformers 5.22.2; huggingface_hub 2.4.0 (full list: `results/followup/env/pip_freeze.txt`, system: `results/followup/env/system.txt`) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (a fresh venv holds the stack above) |
| Local | Python 3 standard library for item selection, cue operators, tests and calibration; numpy for the analysis |

Model revisions were recorded from the download cache when each model was first fetched
(`results/followup/env/model_revisions.jsonl`). To regenerate exactly these weights, pass `revision=` with the hashes above.

## Determinism

- Sampling is seeded per (model, item, cell, sample); see `seed_of` in `scripts/followup_generate.py`. vLLM batching can
  still change floating-point results, so replies are archived and checksummed rather than expected to regenerate
  bit-for-bit. Every cell of a model is generated in the same run, with the same settings.
- The steering arm uses Hugging Face `generate` with hooks; its six conditions (including its own baseline and
  instruction-only condition) are generated together in one engine and compared only with each other.
- Model graders are not deterministic; all labels are archived and every analysis reads the archived labels.

## Steps

```bash
python3 scripts/followup_items.py --write          # frozen item selection (fresh items only)
python3 scripts/followup_render.py                 # every (item, cell) prompt + checksums
python3 -m unittest tests.test_followup_cues -v    # 15 checks: byte identity, single-channel changes, lexicon, names
# on the GPU machine (unattended; status lines in logs/followup/status):
bash scripts/followup_driver.sh                    # smoke -> pilot -> grade pilot -> main (5 models) -> grade main
bash scripts/followup_queue.sh                     # then logs/followup/jobs/*.sh: Llama pilot, steering arm, recognition
# measurement gate and analysis (local):
python3 scripts/followup_calibration.py build --judged <pilot judged files>
python3 scripts/followup_calibration.py gate <human csv> <model grader label files>
python3 scripts/followup_analyze.py --judged <main judged files> --claude <validation labels>
```

Results, labels and figures are added below as each stage completes.
