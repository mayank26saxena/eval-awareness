# Steering at a later layer and in a second model: Qwen2.5-7B layer 20, Qwen3-4B layer 13

**Date:** 2026-10-31 · **Status:** final for this project (10 privacy tasks)

Follow-up to the [layer-10 steering reports](2026-10-30_steering_null_qwen2.5-7b.md). Layer-10 steering of the context direction in Qwen2.5-7B-Instruct moved the downstream context readout but not privacy behavior. This report asks whether that null depends on the layer or the model. It repeats the experiment (1) at layer 20 of Qwen2.5-7B, the layer read by the released NLA, and (2) at layer 13 of Qwen3-4B-Instruct-2507, at the same relative depth as layer 10 in Qwen2.5-7B.

## Summary
- **Qwen2.5-7B, layer 20: a tight null.** Steering at layer 20 shifts the downstream context readout by about twice the natural eval–deploy gap in each direction (layer 24: ±23 against a gap of 11.6). Privacy protection does not move: under sampling the sign-specific effect is 0 pp [−5, +3]. With greedy decoding the context direction (−10 pp [−28, +7]) sits inside the distribution of 10 random directions (range −20 to +17 pp).
- **Qwen3-4B, layer 13: a large, specific effect.** Steering toward "eval" raises protection and steering toward "deploy" lowers it. Under sampling the sign-specific effect is **+51 pp [+33, +68]**; with greedy decoding it is +53 pp [+35, +68], above all 10 random directions (range −7 to +25 pp). This is the first context-specific causal effect in the project.
- **But the Qwen3 lever does not explain a natural gap.** In Qwen3-4B the eval context barely changes behavior: the natural eval–deploy gap in protection is +9.5 pp [−16.5, +35]. Steering moves protection on deployment and evaluation prompts alike (deploy 39% → 95%, eval 55% → 100% from −α to +α). Steering toward "eval" gives short, generic refusals (71 words against 112 unsteered). The direction behaves like a general caution setting rather than a record of being tested.
- **Conclusion:** whether the decoded context direction is causal depends on the model. In Qwen2.5-7B it is inert at both layers tested, although that model shows a +22.5 pp behavioral gap. In Qwen3-4B it strongly controls privacy refusals, although that model shows no reliable behavioral gap. In neither model does the detector's reading account for the observed eval–deploy difference.

## Method

| | |
|---|---|
| Prompts | v3 main grid, privacy tasks, explicit grading absent: 40 prompts (10 tasks × 2 formats × {deploy, eval}), as before |
| Directions | `context_probe` (logistic regression, eval vs deploy, leave-one-task-out) and `random_0..9`, fitted at the steered layer; unit-normalized; scaled by the mean last-token activation norm at that layer (95.8 for Qwen2.5-7B layer 20, 33.0 for Qwen3-4B layer 13) |
| Strength | α = ±0.5, added at all prompt and generated positions |
| Sampled sweep | Unsteered and `context_probe` ±0.5, 10 samples per cell (model default generation settings, seeded per cell); 1,200 responses per model |
| Greedy sweep | Unsteered, `context_probe` and 10 random directions at ±0.5; 920 responses per model |
| Manipulation check | Leave-task-out context probes read out downstream (Qwen2.5-7B layers 24, 27; Qwen3-4B layers 20, 27, 35) |
| Scoring | Qwen3-4B-Instruct-2507 judge, privacy rubric; 0 unparsed verdicts; 0 degenerate responses. For Qwen3-4B the judge grades its own model's responses; a hand check of 12 randomly drawn steered responses (6 at each sign) agreed with all 12 verdicts |
| Statistics | Per-prompt protection probability, paired by prompt, bootstrap 95% CI over tasks (`scripts/steer_analysis.py`) |

## Results

### Manipulation check: shift in the downstream context-probe logit at α = ±0.5

| Model, steered layer | Readout layer | Natural gap | context_probe −0.5 / +0.5 | Random directions (max \|shift\|) |
|---|---|---|---|---|
| Qwen2.5-7B, L20 | L24 | +11.6 | −23.4 / +23.1 | 3.0 |
| | L27 | +11.3 | −17.2 / +17.2 | 2.8 |
| Qwen3-4B, L13 | L20 | +14.2 | −4.8 / +10.5 | 3.2 |
| | L27 | +11.2 | −1.3 / +5.2 | 2.6 |
| | L35 | +10.0 | −0.8 / +4.3 | 2.3 |

In both models the context direction moves the readout well beyond random directions. In Qwen3-4B the shift is asymmetric: the "eval" direction is read out more strongly than the "deploy" direction.

### Sampled responses, 10 per prompt

| | Qwen2.5-7B L20, deploy | Qwen2.5-7B L20, eval | Qwen3-4B L13, deploy | Qwen3-4B L13, eval |
|---|---|---|---|---|
| Unsteered protection | 0.66 | 0.88 | 0.81 | 0.91 |
| context_probe −0.5 | 0.70 | 0.87 | 0.39 | 0.55 |
| context_probe +0.5 | 0.73 | 0.82 | 0.95 | 1.00 |
| **Sign-specific effect** (all 40 prompts) | **0.00 [−0.05, +0.03]** | | **+0.51 [+0.33, +0.68]** | |
| Natural eval − deploy gap | +0.225 [+0.025, +0.425] | | +0.095 [−0.165, +0.35] | |

### Sign-specific effect against 10 random directions (greedy)

| Model | context_probe | Random directions: mean, SD, range | Random ≥ context_probe |
|---|---|---|---|
| Qwen2.5-7B, L20 | −0.10 [−0.28, +0.07] | −0.02, 0.14, [−0.20, +0.17] | 6 / 10 |
| Qwen3-4B, L13 | +0.53 [+0.35, +0.68] | +0.06, 0.10, [−0.07, +0.25] | 0 / 10 |

### Response style and coherence (sampled)

| | Mean NLL under unsteered model | Mean words |
|---|---|---|
| Qwen3-4B unsteered | 0.24 | 112 |
| Qwen3-4B context_probe −0.5 / +0.5 | 0.95 / 1.06 | 142 / 71 |
| Qwen3-4B random directions ±0.5 (greedy) | 0.70 / 0.72 | 105 / 105 |
| Qwen2.5-7B L20 unsteered | 0.40 | 105 |
| Qwen2.5-7B L20 context_probe −0.5 / +0.5 | 0.77 / 0.87 | 119 / 99 |

In Qwen3-4B, "eval"-steered responses are short refusals that cite policy ("I can't assist with that request"). "Deploy"-steered responses are longer and more often hand over the private detail directly. Both remain fluent. In Qwen2.5-7B at layer 20, steering toward "eval" changes the wording of responses: refusal phrases ("I can't", "unable to") fall from 70% to 34% of responses. Whether the private detail is disclosed does not change.

## Interpretation
1. **The layer-10 null in Qwen2.5-7B is not a layer artifact.** At layer 20 the readout shift is about four times larger than at layer 10, and the behavioral interval is about three times narrower. It excludes any sign-specific effect above +3 pp. A representation that mediated the +22.5 pp gap would have shown it.
2. **A decodable context direction can be causally potent.** In Qwen3-4B the same procedure finds a direction whose sign controls privacy protection over a range of about 50 pp, which no random direction approaches.
3. **Potent is not the same as explanatory.** The Qwen3-4B direction raises protection on deployment prompts as much as on evaluation prompts, and the model's natural eval context changes protection by only about 10 pp (not significant). The most economical reading is that the eval-vs-deploy probe in Qwen3-4B picks up a general caution or refusal component that the evaluation prompts slightly activate. That would make it a lever on behavior, but not evidence that the model behaves differently because it recognizes a test.

## Limitations
- One strength (α = 0.5), constant steering vector at all positions, one layer per model in this round.
- Ten privacy tasks; honesty not steered.
- Qwen3-4B is graded by its own model family's judge (hand check 12/12, small).
- The caution interpretation is not tested directly; a refusal benchmark on benign requests under the same steering would test it.
- Steering at Qwen3-4B layer 13 is less coherent than random steering of the same norm (NLL about 1.0 against 0.7).

## Artifacts
- `results/qwen2.5-7b-instruct/`: `steer_v3_pii_L20.jsonl`, `steer_v3_pii_L20_sampled.jsonl` and their `.summary.json`, `steer_check_v3_pii_L20.json`
- `results/qwen3-4b-instruct-2507/`: `steer_v3_pii_L13.jsonl`, `steer_v3_pii_L13_sampled.jsonl` and their `.summary.json`, `steer_check_v3_pii_L13.json`
