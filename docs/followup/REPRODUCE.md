# Reproducing the placeholder-names study

Paper: *What's in a Name? How Placeholder Entities Inflate Measured Privacy in Language Models* (source in
`docs/followup/paper/`). Interactive version: <https://mayank26saxena.github.io/blog/placeholder-names-privacy/>.

The design, decision rules and every dated deviation are in `notes/followup_preregistration.md`, which also lists
SHA-256 checksums of every input, frozen in stages before any output existed (git history shows the order:
`FREEZE1` → `FREEZE2` → `FREEZE3` → `FREEZE4` → `FREEZE5`).

## What is in this repository, and what is on request

| In the repository | On request from the author |
|---|---|
| Item selection, cue operators and rendered prompts (`data/followup/`) | Model replies (about 72,000 generations) |
| All scripts, tests and the blind grading tool | Claude Sonnet labels for every reply, and the validation labels |
| Pre-specified plan with deviations (`notes/followup_preregistration.md`) | Human calibration labels |
| Analysis outputs (`results/followup/analysis/sonnet/`), gate result, environment record | |
| SHA-256 manifest of every input, output, label file and script (`results/followup/MANIFEST.sha256`) | |
| Paper and blog sources, with the numbers exported and checked by script | |

With the replies and labels in place, `python3 scripts/followup_manifest.py --check` verifies them against the manifest.

The source benchmarks are not redistributed in raw form. Download ConfAIde tier 3 (<https://github.com/skywalker023/confAIde>,
MIT) into `data/external/confaide/` and PrivacyLens (<https://github.com/SALT-NLP/PrivacyLens>, MIT) into
`data/external/privacylens/` before re-running item selection.

## Environment

| Component | Version |
|---|---|
| Models (generation) | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`, `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659`, `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71`, `google/gemma-2-9b-it` @ `11c9b309abf73637e4b6f9a3fa1e92e615547819`, `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` (pre-specified check, not reported in the paper) |
| Grader (reported results) | Claude Sonnet 5.5, on every reply, after passing a human agreement gate (deviation 11) |
| Pre-specified automated judge | `Qwen/Qwen2.5-32B-Instruct` @ `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, fp8 at load by vLLM; failed the gate (κ = 0.57), not used for reported results |
| GPU stack | one NVIDIA A40 (48 GB), driver CUDA 13.2; Python 3.12.3; vLLM 0.34.0; torch 2.14.0 (cu130); transformers 5.22.2 (full list: `results/followup/env/pip_freeze.txt`, system: `results/followup/env/system.txt`) |
| Local | Python 3 with numpy and matplotlib (`pyproject.toml`, `uv.lock`); `tectonic` for the paper |

Model revisions were recorded when each model was first fetched (`results/followup/env/model_revisions.jsonl`).

## Determinism

- Sampling is seeded per (model, item, version, sample); see `seed_of` in `scripts/followup_generate.py`. vLLM batching
  can still change floating-point results, so replies are archived and checksummed rather than expected to regenerate
  bit for bit. Every version of a model is generated in the same run, with the same settings.
- Model graders are not deterministic; all labels are archived and every analysis reads the archived labels.

## Steps

```bash
python3 scripts/followup_items.py --write          # frozen item selection (fresh items only)
python3 scripts/followup_render.py                 # every (item, version) prompt + checksums
python3 -m unittest tests.test_followup_cues -v    # byte identity, single-channel changes, lexicon, names
# on the GPU machine (unattended; status lines in logs/followup/status):
bash scripts/followup_driver.sh                    # smoke -> pilot -> main generation (and the pre-specified judge)
bash scripts/followup_queue.sh                     # then logs/followup/jobs/*.sh: steering arm, recognition probe
# human gate (blind pilot sheet, tools/followup_grader.html), then full Sonnet grading
python3 scripts/followup_calibration.py build --judged <pilot judged files>
python3 scripts/followup_sonnet_grade.py build     # blind pages; graded in a separate Claude session
python3 scripts/followup_sonnet_grade.py gate      # kappa >= 0.70 against the human labels to go on
python3 scripts/followup_sonnet_grade.py collect
python3 scripts/followup_sonnet_tables.py          # Sonnet-graded copies of every judged file
# analysis on the Sonnet labels -> results/followup/analysis/sonnet/
python3 scripts/followup_analyze.py --judged results/followup/sonnet_full/judged/main/*.jsonl
python3 scripts/followup_analyze_extra.py recog --recognition <recognition files> --judged <main judged files>
# write-up: every number is exported and checked against the analysis outputs, then typeset
python3 scripts/followup_export_blog.py            # blog data; exits non-zero on any mismatch
python3 scripts/followup_paper_assets.py           # paper numbers.tex and figures
(cd docs/followup/paper && tectonic -X compile main.tex)
# integrity
python3 scripts/followup_manifest.py --check       # SHA-256 of every input, output, label and script
```

## What was run (2026-11-17 to 2026-11-20)

| Stage | Rows |
|---|---|
| Pilot (Qwen, Llama; 12 versions, 60 items, k=4) | 2 × 2,880 |
| Main (Qwen, Llama, Mistral, Gemma; 12 versions, 240 items, k=4) | 4 × 11,520 |
| OLMo check (3 versions; not reported in the paper) | 2,880 |
| Steering arm (Qwen, Mistral; 6 conditions, k=2; results in `results/followup/analysis/sonnet/steer.txt`, not reported in the paper) | 2 × 2,880 |
| Recognition probe (4 models, greedy, one answer per prompt) | 4 × 2,880 |
| Sonnet grading of every main, OLMo and steering reply | 54,720 |

All deviations from the plan are listed, dated, in `notes/followup_preregistration.md`.
