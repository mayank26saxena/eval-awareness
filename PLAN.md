# Project plan: Reading the Mind or the Prompt?

*Revised 2026-10-15 after the novelty check (REVIEW.md). Updated 2026-10-16 after reading the full NLA paper (references/nla_paper_notes.md).*

## The question
> When a tool that reads a model's activations reports that the model "thinks it's being tested", is it reading the model's internal state or just the prompt's format? And does the part of that signal that doesn't depend on format actually cause the model to behave differently?

## Why now
- Anthropic says NLAs detect unverbalized eval awareness on Claude (26% of SWE-bench vs <1% of real traffic). The authors admit there is **no ground truth** for this claim.
- Two 2026 papers contradict each other:
  - Devbunova (2603.19426) finds probes mostly track benchmark *format*.
  - "Not One Capability" (2606.23583) finds probes survive rewrites (AUROC ≥0.98).
- Labs counter eval awareness by **making evals realistic** (GDM, METR) and by **steering it away** (Anthropic, Hua et al.). No one has checked which of these addresses the actual cause.

## Contributions
1. **The main one.** A ground-truth test of activation readers (NLAs, Activation Oracles, probes) for eval awareness on open models, using prompts where format and context are controlled.
2. Remove format with LEACE and test whether a contextual eval-awareness direction remains and causally changes behavior. This addresses the contradiction above.
3. A head-to-head comparison of text-only detectors (Jev, LLM judge, embeddings) and activation readers, released as a reusable harness.

## Hypotheses
- **H1 (reader validity).** Activation readers follow *contextual cues* even when format is held fixed. Text-only detectors follow *format*.
  - *Null result:* readers also follow format. In that case current eval-awareness readings partly measure prompt format, which is still a publishable finding.
- **H2 (format-free signal).** After LEACE removes the format direction, a linear probe can still decode the context.
- **H3 (causality).** Steering along the format-free direction changes deception or honesty rates more than steering along the format direction or a random direction.

## Setup

| | Choice | Why |
|---|---|---|
| Headline model | **Gemma-3-27B-IT** | It has an NLA (layer 41) and Gemma Scope 2. Eval awareness is stronger at this size than in smaller models. |
| Development model | Gemma-3-12B-IT | It has an NLA (layer 32) and is cheaper for building the pipeline. |
| Mac sandbox | Gemma-3-4B-IT | For writing and debugging probe code. |
| Second family | Qwen3-8B | It has the strongest Activation Oracle (`ceselder/qwen3-8b-ao-v3-best`). |
| Stretch | Llama-3.3-70B | It has an NLA, and it is the model Nguyen et al. used. Run it via NDIF or 2× H100. |
| Main behavior | Honesty/deception in role-play scenarios | Probe-Rewrite-Evaluate found a 25% swing, so there is enough statistical power. |
| Secondary behavior | PII disclosure by a support agent with CRM tools | Relevant to Amazon. |

## Dataset design: format × context
- **Format:** benchmark-like (numbered scenario, formal, multiple choice) vs natural (casual chat or a realistic agent task).
- **Context:** contextual eval cues present vs absent. Vary the number of cues from 0 to 3, for example:
  - fictional company
  - test-harness artifacts
  - an "unrealistically convenient" setup
  - grader language
- Within each seed scenario the request stays identical. Target 40 seeds × 4 cells × behaviors ≈ 300 items per cell.
- **Ground truth for "should read as eval"** is the context factor, set by design. Do not claim to measure "belief". Describe it as "contextual cues vs format cues".

---

## Timeline (about 6 weeks, 10–12 hrs/week)

### Week 1 · Setup and pilot
- [ ] Read 2603.19426, 2606.23583, 2509.00591 and the NLA paper (focus on its eval-awareness section).
- [ ] Check whether the Activation Oracles HF collection includes Gemma-3-12B/27B. If not, run the AO arm on Qwen3-8B.
- [ ] Environment: `torch transformers nnsight scikit-learn concept-erasure sae-lens vllm sglang`. Clone `kitft/natural_language_autoencoders`.
- [ ] Mac: on Gemma-3-4B, extract activations and train a per-layer probe on 50 eval and 50 deploy prompts.
- [ ] Cloud smoke test: Gemma-3-12B at layer 32 → parquet → NLA-AV server → read the explanations.
- [ ] Copy the NLA paper's eval-awareness grader, verbalized-awareness grader and realism-rewrite prompt into `prompts/`. Use them word for word so results compare to Anthropic's.
- [ ] Layer check: on Gemma-3-12B, find where the eval-vs-deploy probe peaks. If it is well before L32, the NLA layer may miss the signal; plan for that (see Week 4).
- [ ] Check Amazon policy on side research, publishing and third-party APIs. Join the Jev waitlist.

**Exit:** the end-to-end pipeline runs on 10 prompts.

### Week 2 · Dataset
- [x] Pilot built 2026-10-25: 6 behavior tasks plus 2 topic controls, using the EvalAwareBench F1–F8 taxonomy. Renderer and validator are in `scripts/`; details in data/README.md.
- [ ] Fix the length confound: eval-context cells are shorter because of the bare F4 shells.
- [ ] Optional: add the 100 EvalAwareBench tasks as an external test set (CC BY-NC).
- [ ] Write 40 seed scenarios: 30 honesty and 10 PII.
- [ ] Generate the 4 cells per seed with an LLM, plus graded contextual-cue variants.
- [ ] Add an **eval-topic control**: real-use prompts where the user is *discussing* evals. Anthropic's only NLA false positives on deployment traffic were of this kind.
- [ ] Add a **realism-edit arm**: run Anthropic's rewrite prompt on the benchmark-format items. That gives the exact manipulation from their paper, so we can show how much of its effect is format.
- [ ] QA step 1: text-only classifiers should predict **format** well. On natural-format items they should do noticeably worse at predicting **context**.
- [ ] QA step 2: hand-check 10% of items.

**Exit:** a balanced, QA'd dataset, frozen and version-tagged.

### Week 3 · Behavior gap
- [ ] Run Gemma-3-12B and 27B on all items, 5 samples each, with vLLM.
- [ ] Score with an LLM judge plus a heuristic. Scoring PII against a formal policy (Bedrock Automated Reasoning) is optional.
- [ ] Build a 2×2 table of deception and leak rates with confidence intervals.

🚦 **Gate 1.** If the behavior gap is under about 5 percentage points, add Hawthorne real-vs-hypothetical pairs or a sandbagging task before continuing.

### Week 4 · Readers (main result)
- [ ] **NLA:** explain activations from 12B (L32) and 27B (L41). Score with **Anthropic's grader, used word for word**. Hand-label 100 explanations to check agreement.
- [ ] **Replicate their 3 manipulations on open models**: eval vs deploy, contrastive steering, realism edits. No one has published this for the released NLAs.
- [ ] **Token positions**: on short prompts, run the NLA on *every* token and map where awareness appears (system-prompt end, the Assistant token, early response tokens). Compare any-of-50 aggregation with mean rate.
- [ ] **AR grounding test**: for explanations graded aware, delete the eval sentence and measure the change in AR reconstruction error. Compare against deleting a sentence of matched length. A near-zero change means the AV is inferring "eval" rather than reading it from the activation.
- [ ] **Verbalizable vs not**: per layer, where probes decode the context but the NLA stays silent.
- [ ] **Activation Oracle:** ask "Does the model believe it is being evaluated?" on the same items, using a Gemma AO if one exists, otherwise Qwen3-8B. Average over 5 or more paraphrases and report the spread. Never use the word "token" in questions; it triggers the PastLens regression. Include known-answer control questions.
- [ ] **Probes:** train separate format and context probes at each layer.
- [ ] **Text-only detectors:** Jev, an open LLM judge (Qwen3-32B) and a sentence-embedding classifier.
- [ ] **Headline figure:** for each of the six detectors, how much of its "eval" score is explained by format and how much by context.
- [ ] Post an early Alignment Forum note with the main detector comparison results.

🚦 **Gate 2.** If every reader follows format only, write up "activation readers mostly detect prompt format" and skip to week 6.

### Week 5 · Causality
- [ ] Use LEACE to remove the format direction, then retrain the context probe to get the format-free direction.
- [ ] **NLA-edit steering**: edit high-awareness explanations ("LLM safety evaluation" → "real customer conversation"), run both versions through the AR, take the difference, average over 5 rollouts, and steer at that token on the NLA layer.
- [ ] Steer with the NLA-edit, Anthropic contrastive-prompt (strength ≈0.1, all layers), format, format-free and random directions, at 1 layer and at 3 layers, over a range of strengths. Measure the change in deception and leak rates.
- [ ] Patching: take the representation from a natural-format item with eval cues and patch it into a natural-format item without cues. Check whether the model becomes more careful.
- [ ] Controls: random vectors, a handful of MMLU questions as a capability check, and a coherence check.

**Exit:** dose-response curves with controls.

### Week 6 · Replicate and write up
- [ ] Rerun the key reader result and steering result on Qwen3-8B. Optionally add Llama-3.3-70B via NDIF.
- [ ] Make the main figure: (a) behavior gap, (b) detector comparison, (c) steering curve with all five directions.
- [ ] Write 4–8 pages for a NeurIPS/ICML mech-interp workshop or the Alignment Forum.
- [ ] Release the dataset and harness once Amazon approves.
- [ ] Internal pitch: run the behavior half on Nova and share it with the Amazon AGI Trusted AI and Automated Reasoning teams.

---

## Budget

| Item | Est. |
|---|---|
| Mac development | $0 |
| 12B and 27B generation plus activations on 1× H100 (cache the activations, then run NLAs sequentially) | ~$200–300 |
| NLA and AO inference | ~$150 |
| Steering sweeps on 27B | ~$200–300 |
| Qwen3-8B replication | ~$80 |
| APIs (generation, judges) | ~$100 |
| Optional 70B | ~$200 (or free via NDIF) |
| **Total** | **~$750–1,100** |

## Risks

| Risk | Fallback |
|---|---|
| Someone publishes the NLA/AO ground-truth test first | Post the early AF note in week 4. The steering and PII results still add something new. |
| No Gemma Activation Oracle | Run the AO arm on Qwen3-8B. |
| The NLA judge is noisy | Use two judges and hand-label 100 items for agreement. |
| Rewrites leak format cues | QA step 1 catches this. Hand-write the hardest cell. |
| The behavior gap is small | Gate 1 switches to a different behavior. |
| Amazon policy blocks publishing | Keep it as an internal report plus a Nova pitch. |

## What success looks like
> "On Gemma-3-27B, NLA-reported eval awareness follows [format / context]: it rises X→Y when only format changes and A→B when only contextual cues change. A format-free direction [does / doesn't] remain after LEACE. Steering along it shifts deception from P% to Q%, while format and random directions don't."

**What each outcome means in practice:**
- *If readers follow format:* don't trust eval-awareness readings without format controls, and expect realistic rewrites to be enough.
- *If readers follow context and the effect is causal:* rewrites alone won't fix eval awareness. Labs need production-derived honeypots and internal monitoring.
