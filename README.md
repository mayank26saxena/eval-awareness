# Reading the Mind or the Prompt?

**Validating activation-based detectors of evaluation awareness in open LLMs**

Language models often behave differently when they recognize they are being evaluated. Activation-reading tools (Natural Language Autoencoders, Activation Oracles, linear probes) are now used to detect this *evaluation awareness* even when a model does not verbalize it. This project asks two questions:

1. Do these detectors track **contextual cues** that a setup is an evaluation, or only the **format** of the prompt?
2. Does the format-free signal **causally** change safety-relevant behavior?

The goal is a validated **evaluation-awareness gap** metric: an estimate of how much of a model's safe behavior on a given evaluation depends on it recognizing that it is being tested.

→ Full research statement: [`docs/research_statement.md`](docs/research_statement.md)

## Approach

| Axis | Levels | Cue factors ([EvalAwareBench](https://github.com/aisa-group/decomposing-eval-awareness) taxonomy) |
|---|---|---|
| **Format** | natural prose · benchmark template | F3 register, F7 task format |
| **Context** | deployment · evaluation | F1 placeholder entities, F2 contrived details, F4 bare request, F6 eval scaffolding |
| **Explicit** | absent · present | F8 grading language |

The same request is rendered across all cells. Behaviors are **honesty** (disclosing a known defect under management pressure) and **privacy** (withholding PII from an unverified requester). We then:
- measure behavior per cell, scored by an LLM judge calibrated against hand labels
- train layer-wise probes for each axis, including a "naive" eval-vs-deploy probe trained only where format and context agree, as in most existing datasets, and tested where they disagree
- test whether probes transfer to natural evaluation and deployment transcripts
- evaluate NLA readings against the controlled factors, and steer along the context direction with format, contrastive-prompt and random controls (Activation Oracle and format-free LEACE directions planned)

## Preliminary results

Models: Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507; dataset v3 has 38 behavior tasks.

- **Behavior.** In Qwen2.5-7B, contextual evaluation cues raise privacy protection by **+25 pp** (68% → 93%; 95% CI over tasks [+7.5, +45]). They do not change honesty (−2.7 pp; CI [−12.5, +8.0]).
- **Layer specificity.** A naive eval-vs-deploy probe, trained where format and context agree, tracks **format in early and late layers** and **context only in a narrow mid-network band** (Qwen2.5-7B, L10–11). The layer read by the released Qwen2.5-7B NLA (output of block 20) is format-dominated. *Which layer is probed largely determines what an "eval-awareness probe" measures.*
- **Transfer.** A context probe trained only on controlled prompts separates natural evaluation from deployment transcripts at **AUROC 0.83–0.88** (Qwen2.5-7B, L10). That is above an in-domain baseline (0.79), and it holds under length matching and on English-only prompts. Format probes do not transfer.
- **Model dependence.** Qwen3-4B shows no mid-layer context band, and its transfer is weaker (0.71).
- **NLA readings.** At the response-initial token, the released Qwen2.5-7B NLA never verbalizes evaluation awareness (0 of 622 explanations graded YES with the published grader). Its structural vocabulary tracks format (+27 pp), not context. Linear probes at the same layer do separate evaluation from deployment (AUROC 0.78–0.83 on natural prompts). *Decodable is not the same as verbalized.*
- **NLA position sweep.** Over 11 positions per transcript, including the model's own response, the published grader still finds no context effect (any-of-N YES 2% in both evaluation and deployment contexts). Every YES comes from an explicit grading statement in the prompt. The NLA does track context at the end of the system message, but describes it as a fictional scenario (98% vs 35%, format-invariant), which the rubric does not count as evaluation awareness. Response tokens carry no reading.
- **What the NLA notices.** Measured with validated rubrics, the system-message reading is perceived artificiality: "fictional" or "contrived" in 50% of evaluation-context prompts vs 16% of deployment-context prompts, the same in both formats. Changing one cue at a time, only contrived details produce it alone (+18 pp, 95% CI [+8, +30]). No implicit cue produces evaluation language. Only an explicit grading line does (74–92%).
- **Single cues and behavior.** No single context cue reproduces the +25 pp privacy effect (single-cue effects −5 to +10 pp, none significant). Contrived details, the cue the NLA verbalizes, change neither privacy nor honesty. The clearest task (clinic) changes only when all cues are present.
- **Steering.** Adding the layer-10 context direction moves a downstream context probe by up to ~70% of the natural eval–deploy gap, but privacy protection does not follow: the sign-specific effect is −5 to +15 pp (n.s.), while random directions of the same norm lower protection as much or more. The decodable context representation does not drive the privacy effect at this layer.
- **Caveats.** All models are below 8B, each prompt was run once, and the judge was calibrated on pilot responses. These results guide the next phase; they are not conclusions.

Reports: [`steering (2026-10-29)`](results/reports/2026-10-29_steering_privacy_qwen2.5-7b.md) · [`single-cue behavior (2026-10-27)`](results/reports/2026-10-27_single_factor_behavior_qwen2.5-7b.md) · [`NLA artificiality (2026-10-27)`](results/reports/2026-10-27_nla_artificiality_qwen2.5-7b.md) · [`NLA position sweep (2026-10-27)`](results/reports/2026-10-27_nla_positions_qwen2.5-7b.md) · [`NLA validation (2026-10-26)`](results/reports/2026-10-26_nla_qwen2.5-7b.md) · [`round 2 (2026-10-26)`](results/reports/2026-10-26_round2_qwen.md) · [`pilot (2026-10-25)`](results/reports/2026-10-25_pilot_qwen2.5-7b.md)

## Repository structure

```
├── data/
│   ├── taxonomy.yaml          # factor definitions (F1–F8) and axis grouping
│   ├── tasks/                 # task templates (one YAML per scenario)
│   ├── rendered/              # rendered prompt sets (JSONL)
│   └── external/              # third-party natural transcripts (aware-bench)
├── scripts/
│   ├── render_dataset.py      # templates → prompts (main grid, single-factor flips)
│   ├── validate_dataset.py    # schema checks + format/context separability test
│   ├── generate.py            # model responses (HF transformers; MPS/CUDA/CPU)
│   ├── judge.py               # LLM judge with optional calibration against hand labels
│   ├── single_factor_behavior.py  # per-cue behavior effects with task-level CIs
│   ├── steer.py                   # residual-stream steering with context, format, contrastive and random directions
│   ├── steer_analysis.py          # steering effects, sign-specific contrasts, coherence
│   ├── steer_check.py             # manipulation check: downstream context readout under steering
│   ├── score.py               # keyword baseline scorer
│   ├── probe.py               # layer-wise probes, naive-probe and topic-control analyses
│   ├── probe_transfer.py      # transfer of probes to natural eval/deploy transcripts
│   ├── transfer_confounds.py  # length / language controls for the transfer result
│   ├── extract_positions.py   # activations at prompt-internal and response positions
│   ├── nla_explain.py         # NLA activation verbalizer (published recipe, transformers)
│   ├── nla_setting_rubric.py  # artificiality vs evaluation-framing measures
│   ├── nla_grade.py           # published eval-awareness grader + keyword measures
│   ├── run_pipeline.sh        # end-to-end runner (sleep-safe on macOS)
│   └── show_pairs.py          # side-by-side response viewer
├── prompts/                   # verbatim third-party prompts (e.g. published NLA grader)
├── results/
│   ├── reports/               # dated write-ups
│   └── <model>/               # responses, judgments, probe results
└── docs/
    ├── research_statement.md
    ├── roadmap.md
    ├── related_work.md
    ├── nla_paper_review.md
    ├── compute.md
    └── explainers/            # interactive HTML explainers
```

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -r pyproject.toml

# Dataset
python scripts/render_dataset.py --mode main   --out data/rendered/v2_main.jsonl
python scripts/render_dataset.py --mode single --out data/rendered/v2_single.jsonl
python scripts/validate_dataset.py

# Behavior
python scripts/generate.py --model Qwen/Qwen2.5-7B-Instruct --data data/rendered/v2_main.jsonl
python scripts/judge.py results/qwen2.5-7b-instruct/responses_v2_main.jsonl --data data/rendered/v2_main.jsonl

# Representations
python scripts/probe.py --model Qwen/Qwen2.5-7B-Instruct --data data/rendered/v2_main.jsonl
python scripts/probe_transfer.py --model Qwen/Qwen2.5-7B-Instruct
```

Models up to about 8B run on Apple Silicon (MPS, bf16). Larger models need an 80 GB GPU; see [`docs/compute.md`](docs/compute.md).

## Status

See [`docs/roadmap.md`](docs/roadmap.md). Next: sampled responses on the privacy tasks, steering at other layers and positions against a larger set of random directions, and the AR grounding test.

## Data and licenses

- Task templates and rendered prompts in `data/tasks/` and `data/rendered/` are original to this project. Entities in the "realistic" condition are fictional.
- `data/external/` contains transcripts from [EvalDetectBench / aware-bench](https://huggingface.co/datasets/el7982/aware-bench) (CC BY-NC-SA 4.0). The cue taxonomy follows [EvalAwareBench](https://github.com/aisa-group/decomposing-eval-awareness) (CC BY-NC 4.0).

## References

Related work and resources are in [`docs/related_work.md`](docs/related_work.md).
