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
- (planned) evaluate NLA and Activation Oracle readings against the controlled factors, and steer along format-free directions

## Preliminary results (Qwen2.5-7B-Instruct, pilot)

- **Behavior.** In v1, the eval/deploy gap was driven by a dataset confound: bare prompts removed the user's stake. After fixing this (v2), single cues have little effect. The full combination of contextual cues changes privacy behavior (a medical-appointment request is protected only when all cues are present).
- **Probes.** All three axes are linearly decodable at nearly every layer. A naive eval-vs-deploy probe tracks **format in early layers (2–8)** and **context in middle layers (10–18)**. Late layers, including the layer of the released NLA (L20), are unstable across dataset versions. *Which layer you probe largely determines what an "eval-awareness probe" measures.*
- **Caveats.** The pilot uses 6 behavior tasks, and 7B models show weak evaluation awareness. These results guide the next phase; they are not conclusions.

Details: [`results/reports/2026-10-25_pilot_qwen2.5-7b.md`](results/reports/2026-10-25_pilot_qwen2.5-7b.md)

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
│   ├── score.py               # keyword baseline scorer
│   ├── probe.py               # layer-wise probes, naive-probe and topic-control analyses
│   ├── probe_transfer.py      # transfer of probes to natural eval/deploy transcripts
│   └── show_pairs.py          # side-by-side response viewer
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

See [`docs/roadmap.md`](docs/roadmap.md). Current work: running models on dataset v3 (42 tasks), probe transfer to natural transcripts, and second-family replication.

## Data and licenses

- Task templates and rendered prompts in `data/tasks/` and `data/rendered/` are original to this project. Entities in the "realistic" condition are fictional.
- `data/external/` contains transcripts from [EvalDetectBench / aware-bench](https://huggingface.co/datasets/el7982/aware-bench) (CC BY-NC-SA 4.0). The cue taxonomy follows [EvalAwareBench](https://github.com/aisa-group/decomposing-eval-awareness) (CC BY-NC 4.0).

## References

Related work and resources are in [`docs/related_work.md`](docs/related_work.md).
