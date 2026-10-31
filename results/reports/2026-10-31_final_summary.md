# Final summary: what activation-based detectors of evaluation awareness read, and whether it matters for behavior

**Date:** 2026-10-31 · **Models:** Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507, Qwen3-8B · **Status:** project complete at this scale

This report collects the project's results and states its conclusion. Details are in the dated reports linked under each finding.

## Questions
1. Do activation-based detectors of evaluation awareness (linear probes, a Natural Language Autoencoder) track **contextual cues** that a setup is an evaluation, or only the **format** of the prompt?
2. Does the signal they read **causally** change safety-relevant behavior, so that a detector reading indicates how much behavior on an evaluation depends on the model recognizing it?

## Conclusion
- **Detectors do read context, not only format, but only at the right layer.** In Qwen2.5-7B, a context probe trained on controlled prompts transfers to natural evaluation and deployment transcripts. A naive eval-vs-deploy probe tracks format at most layers, including the layer the released NLA reads.
- **What the detectors read does not explain behavior.** In Qwen2.5-7B, evaluation context raises privacy protection by about 25 pp. Steering the decoded context direction leaves behavior unchanged at two layers, with a tight bound at the later one. In Qwen3-4B the decoded direction does control privacy refusals strongly, and in Qwen3-8B modestly, but in neither does the natural evaluation context reliably change behavior.
- **Implication:** a high evaluation-awareness reading from an activation detector is not, on its own, evidence that a model's behavior on that evaluation depends on recognizing the test. Whether the decoded direction is causal varies by model. It needs to be tested with steering against random-direction controls, and compared with the model's actual behavioral gap.

## Findings

### 1. The behavior gap is real but narrow
- **Qwen2.5-7B (v3, 38 tasks).** Evaluation-context cues raise privacy protection by **+25 pp** (68% → 93%, 95% CI [+7.5, +45]). They do not change honesty (−2.7 pp [−12.5, +8.0]). The gap is concentrated in about five of ten privacy tasks. ([round 2](2026-10-26_round2_qwen.md))
- **Single cues.** No single cue reproduces the effect (−5 to +10 pp, none significant). The effect needs several cues together. ([single-cue behavior](2026-10-27_single_factor_behavior_qwen2.5-7b.md))
- **Qwen3-4B and Qwen3-8B.** Under sampling, the natural privacy gap is +9.5 pp [−16.5, +35] and +12 pp [−4, +34], neither significant. ([steering, layer 20 and Qwen3](2026-10-31_steering_layer20_qwen3.md))

### 2. Probes: context is decodable, but layer choice decides what is measured
- Context is linearly decodable at AUROC 1.0 at every layer of both models on the controlled grid.
- **Naive probe in Qwen2.5-7B.** A naive eval-vs-deploy probe, trained where format and context agree as in most existing datasets, tracks context only in a narrow mid-network band (L10–11) and format elsewhere. That includes the NLA's layer.
- **Naive probe in Qwen3-4B.** It tracks format at nearly every layer.
- **Naive probe in Qwen3-8B.** It shows a context band (L10–18), as in Qwen2.5-7B. The layer pattern therefore holds in two of three models. ([Qwen3-8B](2026-10-31_steering_qwen3-8b.md))
- **Transfer.** The Qwen2.5-7B context probe at L10 separates natural evaluation from deployment transcripts at AUROC 0.83–0.88, above an in-domain baseline (0.79) and robust to length and language controls. Qwen3-4B transfer is weaker (0.71). ([round 2](2026-10-26_round2_qwen.md))

### 3. The NLA does not verbalize evaluation awareness; it notices artificiality
- With the published grader, the released Qwen2.5-7B NLA almost never reports evaluation awareness: 2% of transcripts in both contexts over 11 positions. Every positive comes from an explicit grading line.
- At the end of the system message, the NLA does track context, but it describes it as a fictional or contrived scenario: 50% vs 16% with validated rubrics. Contrived details alone produce that reading (+18 pp [+8, +30]).
- Contrived details do not change behavior. ([NLA validation](2026-10-26_nla_qwen2.5-7b.md), [positions](2026-10-27_nla_positions_qwen2.5-7b.md), [artificiality](2026-10-27_nla_artificiality_qwen2.5-7b.md))

### 4. Steering: the decoded direction does not account for the behavioral gap

Sign-specific effect, protection(+α) − protection(−α), at α = ±0.5 (10 privacy tasks):

| Model, layer | Readout shift (share of natural gap, ±α) | Sign-specific effect, sampled | Context direction vs random directions (greedy) | Natural behavior gap |
|---|---|---|---|---|
| Qwen2.5-7B, L10 | ~45% each way (L20 readout) | −1 pp [−13, +11] | 6 / 20 random ≥ it | +22.5 pp [+2.5, +42.5] |
| Qwen2.5-7B, L20 | ~200% each way (L24 readout) | 0 pp [−5, +3] | 6 / 10 random ≥ it | +22.5 pp [+2.5, +42.5] |
| Qwen3-4B, L13 | −34% / +74% (L20 readout) | **+51 pp [+33, +68]** | 0 / 10 random ≥ it | +9.5 pp [−16.5, +35] |
| Qwen3-8B, L13 | −85% / +77% (L20 readout) | +19 pp [+10, +29] (5 samples) | 0 / 10 random ≥ it (random max +15) | +12 pp [−4, +34] |

- **Qwen2.5-7B: decodable but inert.** The context direction moves the downstream readout strongly, and behavior does not follow. At layer 20, a sign-specific effect larger than +3 pp is excluded, although the natural gap is +22.5 pp.
- **Qwen2.5-7B: protection is fragile.** On evaluation prompts, arbitrary perturbations of the same size lower protection in either sign. ([layer 10](2026-10-29_steering_privacy_qwen2.5-7b.md), [null checks](2026-10-30_steering_null_qwen2.5-7b.md))
- **Qwen3-4B: potent, but not a gap mediator.** The context direction controls protection over about 50 pp, beyond any random direction. It does so on deployment prompts as much as on evaluation prompts, and "eval" steering yields short, generic refusals. The model's natural evaluation context hardly changes behavior. The direction looks like a general caution component that the evaluation prompts load on weakly. ([layer 20 and Qwen3](2026-10-31_steering_layer20_qwen3.md))
- **Qwen3-8B: in between.** The context direction has a modest, one-sided effect: steering toward "deploy" lowers protection (−26 pp on deployment prompts), and steering toward "eval" leaves it unchanged near ceiling. The effect is at the top of the random distribution. As in Qwen3-4B, the natural gap is not significant. ([Qwen3-8B](2026-10-31_steering_qwen3-8b.md))

## Interpretation
1. **What the detector reads.** Linear context probes read something real and transferable about evaluation-like context. The NLA at the tested layer reads perceived artificiality. Neither is the same as a mechanism by which the model changes its behavior when tested.
2. **Two ways a reading can fail as a gap estimate.** In Qwen2.5-7B the representation is decodable but causally inert, while behavior does change with context. In Qwen3-4B the representation is causally potent, while behavior barely changes with context. In either case a detector score would misestimate how much safe behavior depends on recognizing the test. Qwen3-8B falls between the two (a modest causal effect, no significant gap). Over the three models the causal weight of the direction (0, +19, +51 pp) is unrelated to the size of the natural gap.
3. **Method.** Random-direction distributions and sampled responses were necessary throughout. With three random directions and greedy decoding, chance effects of ±15–35 pp would have looked like causal evidence for or against the context direction.

## Limitations
- **Scale.** Models are at most 8B. Evaluation awareness is expected to be stronger in larger models, where the gap and its mechanism may differ.
- **Behaviors.** Two behaviors, with a significant gap only in privacy, over ten tasks.
- **Synthetic cues.** The cues are lexically clear, and probes saturate on them. Subtler cues were not tested.
- **Steering scope.** A single linear direction per layer, a constant vector at all positions, one or two layers per model. LEACE erasure, position-restricted steering and activation patching were not run.
- **Judge.** One local judge (Qwen3-4B-Instruct-2507), calibrated at 47/48 on hand labels. It grades its own model's responses in the Qwen3-4B experiments.
- **Readers.** One NLA, one sample per activation. Activation Oracles were not evaluated.

## Open questions
- Does the pattern hold at 27B–70B, where published NLAs and stronger evaluation awareness exist?
- In Qwen3-4B, is the context direction a refusal direction? A benign-request refusal benchmark under the same steering, and the cosine with a known refusal direction, would tell.
- Does format-free erasure (LEACE) of the context signal remove the privacy gap in Qwen2.5-7B, where single-direction steering does not?
- AR grounding: does ablating the NLA's artificiality claim change reconstruction error?
