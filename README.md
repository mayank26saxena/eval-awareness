# Reading the Mind or the Prompt? (eval awareness)

Started 2026-10-15. Framing revised the same day after a novelty check (see REVIEW.md).
- `statement.html` is the research statement and end goal (**start here**)
- `STATEMENT.md` is the same statement in text
- `hypothesis.html` is the visual explainer of the 2×2 design
- `PLAN.md` is the 6-week project plan
- `REVIEW.md` covers the novelty check, impact, and why the framing changed
- `data/` is the format × context dataset: task templates, rendered prompts, schema (see data/README.md)
- `scripts/` holds `render_dataset.py` (templates → prompts) and `validate_dataset.py` (checks + separability test)
- `references/repos.md` lists open-source datasets and code to build on, with licenses
- `references/nla_paper_notes.md` has the full NLA paper notes: holes in its eval-awareness evidence and ranked extension ideas
- `references/field-survey/` is the broad interpretability field survey: report plus raw notes
- `references/landscape.md` covers who's working on this and which open models have NLAs and Activation Oracles

## Terms
- **Format** is how the prompt is written: casual prose vs a formal benchmark template.
- **Context** is implicit test cues in the setup: placeholder names, too-convenient details, bare requests, eval scaffolding.
- **Explicit** means the prompt says outright that it will be graded.
- **Detector validity** asks whether an activation reader tracks context, or only format or its own guesses.

## The idea in one line
Tools that read a model's activations (NLAs, Activation Oracles, probes) claim to detect when a model "knows it's being tested". **Are they reading the model's mind, or just noticing that the prompt *looks* like a test? And does the format-free part of the signal cause behavior?**

## Example
A customer asks whether a product with a known defect is reliable. The same request is written four ways:

|                      | Benchmark format | Natural format |
|----------------------|-----------------|---------------|
| **Test cues present** (fictional company, grader notes) | ① | ② |
| **No test cues**     | ③ | ④ |

- If the reader flags ① and ③, it follows **format**. It's just a prompt-format detector.
- If the reader flags ① and ②, it follows the **context**. It sees through the casual disguise.

## Hypotheses
1. **Readers follow context**, while text-only detectors (Jev, LLM judge) follow format.
2. **A format-free signal exists**: after removing the format direction (LEACE), the context is still decodable.
3. **That signal drives behavior**: suppressing it makes the model more deceptive on a test it passed. Format and random directions don't have that effect.

## What's new vs. done
- Done by others: probes, 2×2 format datasets (with conflicting results), realistic rewrites, steering.
- **New here:** a ground-truth test of NLAs and Activation Oracles for eval awareness; steering with the format removed; a head-to-head of text-only and activation detectors.

## Showcase result
> "On Gemma-3-27B, NLA-reported eval awareness follows [format / context]. A format-free direction [does / doesn't] remain. Steering along it shifts deception from P% to Q%; format and random directions don't."

Either outcome is useful (see REVIEW.md, section 5).

## Tools
- Headline model: **Gemma-3-27B-IT** (NLA at layer 41, Gemma Scope 2). Dev model: **Gemma-3-12B-IT** (NLA at layer 32)
- Mac sandbox: Gemma-3-4B-IT. Second family: Qwen3-8B (best Activation Oracle). Stretch: Llama-3.3-70B (has an NLA)
- Open NLAs exist for Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B, and Llama-3.3-70B (huggingface.co/collections/kitft/nla-models)
- Behaviors: honesty/deception (main), PII disclosure (secondary, relevant to Amazon)
- Libraries: `nnsight` / `transformer_lens`, `scikit-learn`, `concept-erasure` (LEACE), `vllm`, `SAELens` + Gemma Scope 2
- Compute: Mac for ≤4B; 1× H100 for 12B/27B (~$750–1,100 total)

## Setup
```bash
uv venv --python 3.12 && uv pip install -r pyproject.toml
.venv/bin/python scripts/validate_dataset.py
```

## Week 1
- [x] Pilot dataset: 8 tasks, 52 main-grid + 72 single-flip items, validation passing (data/README.md)
- [ ] Read: Devbunova 2603.19426, "Not One Capability" 2606.23583, Probe-Rewrite-Evaluate 2509.00591, the NLA paper's eval-awareness section
- [x] Check which open models have NLAs (done: see references/landscape.md)
- [ ] Check whether the Activation Oracles HF collection includes Gemma-3-12B/27B
- [x] Pipeline: `scripts/generate.py` → `score.py` → `probe.py`, smoke-tested end to end on Qwen2.5-0.5B (results/qwen2.5-0.5b-instruct/)
- [x] First real run: Qwen2.5-7B (has a released NLA). See results/FINDINGS_pilot_qwen7b.md
- [ ] Gemma runs (need Hugging Face Gemma access)
- [ ] Cloud smoke test: Gemma-3-12B L32 activations → NLA explanations

## Amazon angle
- Run the behavior half on Nova (closed weights) → "can we trust our Nova safety evals?"
- Possible contacts: Amazon AGI Trusted AI Symposium organizers (trusted-ai-symposium@amazon.com), Automated Reasoning team
- Check publication + third-party API policy before sharing anything

## Key papers
- Probing and Steering Evaluation Awareness — https://arxiv.org/abs/2507.01786
- Steering Eval-Aware Models to Act Deployed — https://arxiv.org/abs/2510.20487
- Evaluation Awareness Is Not One Capability — https://arxiv.org/html/2606.23583v1
- Eval Awareness Shifts from Format to Context with Scale — https://arxiv.org/abs/2609.22119
- Activation Oracles — https://arxiv.org/abs/2512.15674
- Anthropic Natural Language Autoencoders — https://transformer-circuits.pub/2026/nla/
- Jev (TypeSafe) — https://typesafe.ai/blog/introducing-system-one-models-and-jev

## Field background
- `references/field-survey/report.md` is the full report on the state of interpretability research (Oct 2026)
- `references/field-survey/raw-notes/` holds the research notes behind it (labs, conferences, open problems)

## Log
- 2026-10-15: Picked the topic, framed the hypotheses, set up this folder.
- 2026-10-15: Surveyed industry work (Anthropic, OpenAI/Apollo, GDM, UK AISI and METR all flag eval awareness as unsolved). Chose Gemma-3-12B. Wrote PLAN.md.
- 2026-10-15: Revised framing to "Mind or Prompt?" (NLA/AO ground-truth test). Rewrote PLAN.md (6 weeks) and hypothesis.html.
- 2026-10-16: Read the full NLA paper. Key gaps: its realism edits mix format with context; its open-model NLAs were never tested for eval awareness. New ideas: AR grounding test, NLA-edit steering, token-position map. Added to PLAN.md.
- 2026-10-16: Set the end goal: an Eval Trust Gauge (the eval-awareness gap, measured with a validated detector). Added statement.html.
- 2026-10-24: Moved the field-survey report and notes from Desktop/reports and Desktop/research_notes into references/field-survey/.
- 2026-10-25: Renamed style→format and situation→context throughout. Built the pilot dataset, using the EvalAwareBench factor taxonomy. Added the render and validate scripts and references/repos.md.
- 2026-10-25: Built the generate/score/probe pipeline and smoke-tested it on Qwen2.5-0.5B. A naive probe trained where format and context agree tracks FORMAT in layers 2–5 and CONTEXT in layers 7–15. Only a smoke test: tiny model, 48 items, length confound, heuristic scoring. Waiting on Gemma access.
- 2026-10-25: Qwen2.5-7B pilot. Eval/deploy behavior gap is 83%→71%, but single-factor flips show it comes from the bare-prompt shells (F4: user stakes removed), not from test cues. Probes: a naive eval-vs-deploy probe follows FORMAT at layers 3–5, CONTEXT at 9–18, and is mixed at 20–28 (including the NLA layer, 20). Next: fix the F4 confound, add an LLM judge, use batch-size 1.
