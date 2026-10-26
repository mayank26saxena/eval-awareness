# Research statement

**Title:** *Reading the Mind or the Prompt? Validating activation-based detectors of evaluation awareness in open LLMs*

Visual version: `statement.html`

## End goal
Build and validate an **Eval Trust Gauge**: for any safety eval, measure **how much of the model's good behavior exists only because it knows it's being tested**.

Gap = misbehavior with the "being tested" signal switched off − misbehavior normally.

The gauge only works if three claims hold:
1. the detector reads the model, not the prompt
2. the reading is grounded, not invented
3. the signal causes the behavior

The research questions below test those three claims.

**Using the gap:** audit evals, find the cues that give tests away, flag test-aware transcripts, and point red-teamers at fragile spots. Two uses to avoid:
- Don't keep the model "always feeling watched" in deployment. That hides the problem.
- Don't train the signal away. The model may just learn to hide it.

## One sentence
Tools that read a model's activations (Natural Language Autoencoders, Activation Oracles, probes) are now used to claim that models *silently* know when they're being tested. I test whether those readings reflect the model's internal state or just how the prompt looks, and whether the signal they find actually changes the model's behavior.

## Problem
Safety evals only work if models behave the same way when tested as in deployment. Anthropic's NLA paper (May 2026) reports *unverbalized* eval awareness in Claude and plans to use NLAs in pre-deployment audits. Its validation has three weaknesses:
- **It is only directional.** There is no ground truth.
- **It mixes two things.** The "realism edits" change prompt format and contextual cues at the same time.
- **It covers only Claude.** The released open-model NLAs were never tested for eval awareness.

Probe studies disagree with each other. One finds probes track *format*; another finds they survive format changes.

## Research questions
- **RQ1 – Validity:** Do activation readers track **contextual cues** (signs the setup is an evaluation) or **format cues** (how the prompt looks)?
- **RQ2 – Grounding:** When an NLA says "the model thinks this is an eval", is that information present in the activation, or did the explainer infer it?
- **RQ3 – Causality:** Does a format-free eval-awareness direction causally change safety-relevant behavior?

## Hypotheses
- **H1.** Text-only detectors follow format. Activation readers follow context even when format is held fixed.
- **H2.** Eval-awareness claims in NLA explanations measurably contribute to reconstructing the activation, so they aren't confabulated.
- **H3.** Suppressing the format-free direction increases deception or PII leakage on evals the model passes. Format and random directions don't.

## Method (Gemma-3-27B-IT headline · Gemma-3-12B dev · Qwen3-8B replication)
1. **Controlled dataset:** format (benchmark vs natural) × context (test cues vs none), plus a real-use eval-topic control. Behaviors are honesty/deception (main) and PII disclosure (secondary).
2. **Detector panel:**
   - activation readers: NLA (scored with Anthropic's own grader), Activation Oracle, layer-wise probes
   - text-only baselines: Jev, LLM judge, embedding classifier
   - for each detector, measure how much of its signal comes from format vs context
3. **Grounding test:** delete eval-awareness claims from NLA explanations and measure the change in reconstruction error.
4. **Causal test:** steer along four directions and measure behavior change: the format-free (LEACE) direction, an NLA-edit direction, Anthropic's contrastive vector, and a random direction.

## Contributions
1. The first ground-truth validation of NLA and Activation Oracle eval-awareness readings, run on open models.
2. A test that separates grounded readings from inferred ones, using the NLA's own reconstructor.
3. Causal evidence for or against a format-free eval-awareness signal, which would settle the disagreement between the probe studies.
4. A reusable, format-controlled dataset and detector harness.

## Why it matters, either way
- **If readers follow format:** eval-awareness readings need format controls before anyone trusts them, and realistic rewrites may be enough.
- **If they follow context and the effect is causal:** rewrites won't fix the problem. Labs need production-derived evals and internal monitoring, and this project yields a validated monitor plus an "eval-awareness gap" metric that could be applied to models like Nova.

**Scope:** about 6 weeks part-time, roughly $750–1,100 of compute. Target: NeurIPS/ICML mech-interp workshop or the Alignment Forum.
