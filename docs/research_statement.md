# Research statement

**Reading the Mind or the Prompt? Validating activation-based detectors of evaluation awareness in open LLMs**

An interactive version is available at [`explainers/research_statement.html`](explainers/research_statement.html).

## Summary
Activation-reading tools (Natural Language Autoencoders, Activation Oracles, linear probes) are increasingly used to claim that language models internally recognize when they are being evaluated, even when they do not say so. This project tests whether those readings reflect the model's internal state or only the surface format of the prompt, and whether the underlying signal causally changes the model's behavior.

## Motivation
Safety evaluations assume a model behaves the same way under test as in deployment. Frontier-lab reports document that models often recognize evaluations and behave differently when they do (see [`related_work.md`](related_work.md)). Recent work reports *unverbalized* evaluation awareness detected with NLAs, but its validation has three limitations:
- **Directional only.** There is no ground truth for the model's belief.
- **Confounded manipulation.** The "realism edits" change prompt format and contextual cues at the same time.
- **Single model family.** The released open-model NLAs were not evaluated for eval awareness.

Probe-based studies also disagree on whether eval-awareness probes track format or something deeper.

## Objective
Develop and validate an **evaluation-awareness gap** metric. For a given evaluation, it estimates how much of a model's safe behavior depends on its internal representation of being evaluated:

> gap = misbehavior rate with the evaluation-awareness signal suppressed − misbehavior rate unmodified

The metric is meaningful only if the three research questions below are answered affirmatively.

## Terminology
- **Format:** surface presentation of the prompt, for example casual prose vs a benchmark-style template.
- **Context:** implicit cues that the setup is an evaluation, such as placeholder entities, overly convenient details, missing surrounding context, or evaluation scaffolding.
- **Explicit:** text stating that the response will be graded.
- **Detector validity:** whether a detector's eval-awareness reading tracks context, rather than format or the detector's own inferences.

## Research questions and hypotheses

| | Question | Hypothesis |
|---|---|---|
| **RQ1 Validity** | Do activation-based detectors track contextual cues or format cues? | Text-only detectors follow format. Activation-based detectors follow context when format is held fixed. |
| **RQ2 Grounding** | When an NLA reports evaluation awareness, is that information present in the activation? | Eval-awareness claims measurably contribute to AR reconstruction, so they are not confabulated. |
| **RQ3 Causality** | Does a format-free eval-awareness direction causally affect safety-relevant behavior? | Suppressing it increases deception or PII disclosure. Format and random directions do not. |

## Method
1. **Controlled dataset.** Format × context × explicit, using the EvalAwareBench cue taxonomy, plus a control for real use that discusses evaluations. Behaviors are honesty (disclosing a known defect) and privacy (withholding PII). See [`../data/README.md`](../data/README.md).
2. **Detector panel.** Activation-based: linear probes at every layer, NLA (scored with the published grader), Activation Oracle. Text-only: LLM judge, embedding classifier. Each detector's signal is decomposed into format and context components.
3. **Grounding test.** Ablate eval-awareness claims from NLA explanations and measure the change in reconstruction error.
4. **Causal test.** Steer along four directions: format-free (LEACE), NLA-edit, contrastive-prompt and random. Measure the change in behavior.
5. **Transfer.** Evaluate probes trained on controlled data on natural evaluation and deployment transcripts.

Models: Gemma-3-27B-IT (main), Gemma-3-12B-IT, Qwen2.5-7B-Instruct and Qwen3-4B-Instruct (development and replication). All have released NLAs or Activation Oracles, except Qwen3-4B.

## Expected contributions
1. A ground-truth validation of NLA and Activation Oracle eval-awareness readings on open models.
2. A grounding test that separates activation-supported readings from explainer inference.
3. Causal evidence on whether a format-free eval-awareness signal exists.
4. A format-controlled dataset and an open detector-evaluation harness.

## Implications
- **If detectors follow format:** eval-awareness readings require format controls, and more realistic evaluation design may be sufficient.
- **If detectors follow context and the effect is causal:** realism rewrites are insufficient. Evaluations should be complemented by production-derived scenarios and internal monitoring, and the gap metric provides a way to audit them.

## Intended uses and non-uses
The gap metric is a **diagnostic**. It is not intended for keeping an evaluation-awareness signal active in deployment, or as a training target. Training against it risks teaching the model to hide the representation.
