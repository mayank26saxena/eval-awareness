# How test design changes what LLM evaluations measure

Code, data and write-ups for two studies of open language models (up to 9B parameters). Each asks whether a safety
evaluation measures the model's behavior or an artifact of how the test is written.

| Study | Question | Write-up |
|---|---|---|
| [**Placeholder names and measured privacy**](#1-placeholder-names-and-measured-privacy-november-2026) (Nov 2026) | Does writing people as "Person A" change privacy scores? | [Paper source](docs/followup/paper/) · [Interactive post](https://mayank26saxena.github.io/blog/placeholder-names-privacy/) |
| [**Validating evaluation-awareness detectors**](#2-validating-evaluation-awareness-detectors-october-2026) (Oct 2026) | Do activation-based "is this a test?" detectors read evaluation context, and does it drive behavior? | [Final summary](results/reports/2026-10-31_final_summary.md) |

The two studies are independent: the first does not use any results from the second.

---

## 1. Placeholder names and measured privacy (November 2026)

*What's in a Name? How Placeholder Entities Inflate Measured Privacy in Language Models*

Privacy benchmarks write people either with realistic names or with placeholders such as "Person A", and the choice is
usually treated as neutral. In a pre-specified, item-paired study of four open models (Qwen2.5-7B, Llama-3.1-8B,
Mistral-7B, Gemma-2-9B) on 240 items from ConfAIde and PrivacyLens:

- **Placeholders raised measured protection** (the share of replies that withhold the private detail) by **+3.7
  percentage points** (95% CI [+1.4, +6.0]; Holm-adjusted p = 0.006), with positive estimates in all four models.
- **Realistic renaming did not** (−0.2, [−1.8, +1.4]), on one draw of alternative names.
- **The effect is carried by ConfAIde's narrative stories** (+6.4, [+4.3, +8.5]). On PrivacyLens agent tasks the
  estimate is inconclusive (+1.1, [−1.4, +3.7]).
- For scale, a one-sentence privacy instruction in the system prompt added +11.0 points; test-like formatting and a
  scoring note had little effect.

All 46,080 replies were graded by Claude Sonnet after it passed an agreement gate against human labels (κ = 0.93).

| What | Where |
|---|---|
| Pre-specified plan, decision rules, dated deviations | [`notes/followup_preregistration.md`](notes/followup_preregistration.md) |
| Reproduction guide, and what is available on request | [`docs/followup/REPRODUCE.md`](docs/followup/REPRODUCE.md) |
| Items, name list, rendered prompts (cue operators: `scripts/followup_cues.py`) | [`data/followup/`](data/followup/) |
| Pipeline: selection, rendering, generation, grading, analysis | `scripts/followup_*.py`, tests in `tests/test_followup_*.py` |
| Analysis outputs (Sonnet-graded), checksum manifest, environment | [`results/followup/`](results/followup/) |
| Paper (LaTeX) and blog source; every number exported and checked by script | [`docs/followup/paper/`](docs/followup/paper/), [`docs/followup/blog/`](docs/followup/blog/) |

Model replies and labels are not in the repository; they are available from the author on request.

---

## 2. Validating evaluation-awareness detectors (October 2026)

*Reading the Mind or the Prompt?*

Activation-reading tools (linear probes, Natural Language Autoencoders) are used to detect when a model recognizes it is
being evaluated. This study renders the same requests with independent switches for **format** (natural prose vs
benchmark template), **context** (deployment vs evaluation cues) and **explicit** grading language, and asks whether
detectors track context or just format, and whether what they read drives behavior. Models: Qwen2.5-7B-Instruct,
Qwen3-4B-Instruct-2507 and Qwen3-8B.

**Conclusion.** Detectors do read evaluation context, but only at the right layer, and what they read does not explain
the behavioral difference between evaluation and deployment. A detector reading alone is not evidence that behavior on
an evaluation depends on recognizing the test.

- **Behavior.** In Qwen2.5-7B, evaluation cues raise privacy protection by +25 pp (68% → 93%; CI [+7.5, +45]) and do
  not change honesty.
- **Layer matters.** A naive eval-vs-deploy probe tracks format in early and late layers and context only in a narrow
  mid-network band (Qwen2.5-7B, layers 10–11).
- **Transfer.** A context probe trained on controlled prompts separates natural evaluation from deployment transcripts
  (AUROC 0.83–0.88); format probes do not transfer.
- **NLA readings.** The released Qwen2.5-7B NLA never verbalizes evaluation awareness at the response token (0 of 622).
  What it does pick up is perceived artificiality ("fictional", "contrived"), driven by contrived details.
- **Steering.** Steering along the context direction moves a downstream probe but not privacy behavior in Qwen2.5-7B.
  In Qwen3-4B it raises refusals on deployment prompts as much as on evaluation prompts, acting as a general caution
  lever rather than a mediator.

Details and caveats: [final summary](results/reports/2026-10-31_final_summary.md), dated reports in
[`results/reports/`](results/reports/), and the [research statement](docs/research_statement.md).

```bash
uv venv --python 3.12 && uv pip install -r pyproject.toml
python scripts/render_dataset.py --mode main --out data/rendered/v3_main.jsonl && python scripts/validate_dataset.py
python scripts/generate.py --model Qwen/Qwen2.5-7B-Instruct --data data/rendered/v3_main.jsonl
python scripts/judge.py results/qwen2.5-7b-instruct/responses_v3_main.jsonl --data data/rendered/v3_main.jsonl
python scripts/probe.py --model Qwen/Qwen2.5-7B-Instruct --data data/rendered/v3_main.jsonl
```

Models up to about 8B run on Apple Silicon (MPS, bf16); see [`docs/compute.md`](docs/compute.md).

---

## Repository layout

```
data/
  followup/          study 1: item manifest, name list, rendered prompts
  tasks/, rendered/  study 2: task templates and prompt sets (taxonomy.yaml: cue definitions)
  external/          study 2: natural transcripts from aware-bench
scripts/
  followup_*         study 1 pipeline
  everything else    study 2 (render, generate, judge, probe, NLA, steer)
results/
  followup/          study 1 analysis outputs, manifest, environment record
  reports/           study 2 dated reports
  <model>/           study 2 responses, judgments, probe results
docs/
  followup/          study 1 paper, blog source, REPRODUCE.md
  *.md, explainers/  study 2 research statement, related work, roadmap
notes/               study 1 pre-specified plan
tools/               blind grading page used for the human labels
```

## Data and licenses

Code is released under the [MIT License](LICENSE). Data keep their own licenses:

- `data/followup/` prompts are derived from [ConfAIde](https://github.com/skywalker023/confAIde) and
  [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens) (both MIT). The raw benchmarks are not redistributed; see
  [`REPRODUCE.md`](docs/followup/REPRODUCE.md) for downloading them.
- `data/tasks/` and `data/rendered/` are original to this project; all people in them are fictional.
- `data/external/` holds transcripts from [aware-bench](https://huggingface.co/datasets/el7982/aware-bench)
  (CC BY-NC-SA 4.0). The cue taxonomy follows [EvalAwareBench](https://github.com/aisa-group/decomposing-eval-awareness)
  (CC BY-NC 4.0).
