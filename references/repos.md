# Open-source code and data to build on

Checked 2026-10-25. **Licenses matter.** Several of these are non-commercial, so check with Amazon before reusing code or data at work.

## Datasets

| Resource | What it is | Use for us | License / access |
|---|---|---|---|
| **EvalAwareBench** · [github.com/aisa-group/decomposing-eval-awareness](https://github.com/aisa-group/decomposing-eval-awareness) · HF `aisa-group/EvalAwareBench` · arXiv 2605.23055 | 100 paired safety/capability tasks as YAML templates. **8 eval-cue factors (F1–F8), each switchable independently** while the request stays fixed. Measures only black-box CoT verbalization (API models). | **Our factor taxonomy is borrowed from it.** It also gives a ready-made external test set: run our activation readers on their factor toggles. Their safety tasks are HarmBench-style refusal prompts. | CC BY-NC 4.0 (non-commercial) |
| **Needham et al. eval_awareness** · [huggingface.co/datasets/jjpn2/eval_awareness](https://huggingface.co/datasets/jjpn2/eval_awareness) · github.com/jjpn97/eval_awareness · arXiv 2505.23836 | 1,000 real transcripts from 61 datasets, labeled eval vs deployment and chat vs agentic. | Natural (not synthetic) eval vs deploy data for probe training and NLA replication. | Gated on Hugging Face; mixed source licenses |
| **EvalDetectBench / aware_bench** · github.com/freeze-lasr/aware_bench · HF `el7982/aware-bench` · arXiv 2609.01611 | 746 trajectories: 413 eval, 333 deploy, chat and agentic. | A second natural eval vs deploy set for generalization checks. | Check the repo |
| **baceolus/eval_awareness** · [github.com/baceolus/eval_awareness](https://github.com/baceolus/eval_awareness) | Black-box tool that finds which prompt features make a model think it's in an eval, then clusters them into a taxonomy. | Optional: check that our cues are the ones models actually notice. | No license file, so don't copy code |

## Code

| Resource | What it is | Use for us | License |
|---|---|---|---|
| **Hua et al. steering** · [github.com/tim-hua-01/steering-eval-awareness-public](https://github.com/tim-hua-01/steering-eval-awareness-public) · arXiv 2510.20487 | Contrastive steering vectors ("you are in evaluation" vs "deployment"), layer sweeps, vLLM generation, Gaussian control vectors. Uses Nemotron-49B plus eval-aware model organisms (`timhua/wood_v2_sftr4_filt`). | Reference implementation for the steering experiments (week 5): contrastive vectors and random-vector controls. | MIT |
| **NLA** · [github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) · HF `kitft/nla-models` | Training and inference code, plus released AV/AR checkpoints for Qwen2.5-7B, Gemma-3-12B/27B, Llama-3.3-70B. SGLang server. | NLA readings, the AR grounding test, NLA-edit steering. | Apache-2.0 (checkpoints inherit the base-model license) |
| **Activation Oracles** · github.com/adamkarvonen/activation_oracles · HF `adamkarvonen/activation-oracles`, `ceselder/qwen3-8b-ao-v3-best` | Trained oracles for Gemma-2/3, Qwen3 and Llama-3, plus an improved Qwen3-8B oracle. | Activation Oracle detector arm. | Check the repo |
| **concept-erasure (LEACE)** · github.com/EleutherAI/concept-erasure | Removes a linearly encoded concept (here, format) from activations. | Building the format-free direction. | MIT |
| **nnsight** · github.com/ndif-team/nnsight | Activation extraction and intervention on any Hugging Face model. NDIF also gives remote access to large models. | Main library for activations, probes and steering. | MIT |
| **TransformerLens** · github.com/TransformerLensOrg/TransformerLens | Hooks and caching for supported models. | Alternative to nnsight for smaller models. | MIT |
| **SAELens + Gemma Scope 2** · github.com/jbloomAus/SAELens | Pretrained SAEs and transcoders for all Gemma 3 models. | Feature-level look at the format and context directions. | MIT |

## Papers with methods to copy
- **Anthropic NLA paper appendix:** the eval-awareness grader, the verbalized-awareness grader, the realism-rewrite prompt and the contrastive steering recipe (see `nla_paper_notes.md`).
- **"The Hawthorne Effect in Reasoning Models"** (arXiv 2505.14617): evaluates and steers test awareness in reasoning models. A useful comparison for the chain-of-thought timing idea.

## Novelty note
EvalAwareBench already varies eval cues factor by factor, but measures them **only from outside the model** (does the chain of thought mention a test?). Our addition is unchanged:
- **activation-based detectors**, checked against those controlled factors
- the **grounding test** on NLA claims
- **causal steering**

Cite EvalAwareBench as the source of the factor taxonomy.
