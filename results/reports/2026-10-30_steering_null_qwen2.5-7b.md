# Steering null: random-direction distribution and sampled responses, Qwen2.5-7B-Instruct, v3

**Date:** 2026-10-30 · **Status:** preliminary (one layer, 10 privacy tasks)

Follow-up to the [layer-10 steering report](2026-10-29_steering_privacy_qwen2.5-7b.md). That report found no sign-specific effect of the context direction on privacy protection, against three random directions and greedy decoding. This report tests whether the null holds with two changes: 20 random directions, to place the context directions within the distribution of arbitrary perturbations, and 10 sampled responses per prompt, to measure disclosure probabilities rather than single outcomes.

## Summary
- **The context directions are typical of random directions.** At α = ±0.5, the sign-specific effect, protection(+α) − protection(−α), is +15 pp for the context probe and −5 pp for the contrastive-prompt direction. Over 20 random directions it ranges from −17 to +35 pp (mean +4, SD 14). 30% and 75% of random directions, respectively, score at least as high as the two context directions.
- **Single greedy outcomes are noisy enough to mislead.** 7 of the 20 random directions have a sign-specific effect whose 95% CI excludes zero (5 positive, 2 negative). With fewer controls, one context direction could have appeared significant by chance.
- **With sampling, the context effect is close to zero and bounded.** With 10 samples per response, the sign-specific effect is −1 pp [−13, +11] for the context probe and −3 pp [−15, +9] for the contrastive-prompt direction.
- **Full mediation is excluded.** From −α to +α, the steering spans about 90% of the natural eval–deploy gap in the layer-20 context readout. The behavioral eval–deploy gap under sampling is +22.5 pp [+2.5, +42.5]. A representation that fully mediated the gap would therefore predict about +20 pp, above both upper bounds. Partial mediation of up to about half the gap is not excluded.

## Method

| | |
|---|---|
| Prompts | As before: v3 main grid, privacy tasks, explicit grading absent, 40 prompts (10 tasks × 2 formats × {deploy, eval}) |
| Random distribution | `random_0..19`, seeded Gaussian directions (the first three are those of the earlier report), α = ±0.5, greedy; 1,360 new responses added to `steer_v3_pii_L10.jsonl` |
| Sampled sweep | Unsteered, `context_probe` and `context_mean` at α = ±0.5; 10 samples per cell with the model's default generation settings (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05), seeded per cell; 2,000 responses (`steer_v3_pii_L10_sampled.jsonl`, `scripts/steer.py --samples 10`) |
| Scoring | Same judge and rubric as before; 0 unparsed verdicts; 0 degenerate responses in the sampled sweep |
| Statistics | Sampled responses are averaged per prompt (protection probability), then paired and bootstrapped over tasks as before. Random-null rank: fraction of the 20 random directions whose sign-specific effect is at least that of the given direction (`scripts/steer_analysis.py`) |

Why α = 0.5: it shifts the downstream context readout substantially (about 45% of the natural gap at layer 20 in each direction), while keeping responses coherent (mean NLL 0.5–0.75 under the unsteered model, against 0.3 unsteered).

## Results

### Sign-specific effect at α = 0.5 against 20 random directions (greedy)

| Direction | protect(+α) − protect(−α) | 95% CI | Random directions ≥ this |
|---|---|---|---|
| context_probe | +0.15 | [−0.10, +0.40] | 6 / 20 |
| context_mean | −0.05 | [−0.25, +0.12] | 15 / 20 |
| format_probe | +0.03 | [−0.17, +0.20] | 13 / 20 |
| Random (20) | mean +0.04, SD 0.14, range [−0.17, +0.35] | | |

Change from unsteered, mean over the 20 random directions (range):

| | Deploy, +0.5 | Deploy, −0.5 | Eval, +0.5 | Eval, −0.5 |
|---|---|---|---|---|
| Random | −0.04 [−0.20, +0.10] | −0.08 [−0.25, +0.10] | −0.13 [−0.35, 0.00] | −0.18 [−0.45, 0.00] |
| context_probe | +0.10 | 0.00 | −0.10 | −0.30 |
| context_mean | 0.00 | 0.00 | −0.25 | −0.15 |

On evaluation-context prompts (unsteered protection 90%), nearly every direction lowers protection, in either sign.

### Sampled responses, 10 per prompt

| | Deploy prompts | Eval prompts |
|---|---|---|
| Unsteered protection | 0.66 | 0.88 |
| context_probe, −0.5 / +0.5 | −0.02 / +0.01 | −0.15 / −0.19 |
| context_mean, −0.5 / +0.5 | −0.07 / −0.09 | −0.14 / −0.18 |

| Direction | Sign-specific effect | 95% CI |
|---|---|---|
| context_probe | −0.01 | [−0.13, +0.11] |
| context_mean | −0.03 | [−0.15, +0.09] |

The eval–deploy gap under sampling is +0.225 [+0.025, +0.425], matching the greedy estimate (+0.25). Per task it ranges from −0.40 (tenant) to +0.70 (clinic).

## Interpretation
1. **The steering null is robust.** It holds against a 20-direction random distribution and with sampled responses, which give intervals about half as wide as greedy decoding. Steering the context representation at layer 10 in the "eval" direction is no different from steering in the "deploy" direction.
2. **What remains possible.** The bounds exclude the representation fully mediating the privacy gap, but not a partial contribution of up to about half of it. They also say nothing about other layers, about steering confined to particular positions, or about a nonlinear or distributed encoding that a single direction misses.
3. **Protection is asymmetric under perturbation.** On evaluation-context prompts, steering in either sign lowers protection by 15–19 pp under sampling. Protection in this setting looks like a fragile state that perturbations knock the model out of, rather than a quantity read off a context variable.

## Limitations
- One layer, one constant steering vector at all positions, one strength for the random distribution.
- Ten privacy tasks; the behavioral gap is concentrated in about five of them.
- One model.

## Next steps
1. **Steering at other layers** (14, 20) and only at the system-message tokens, with sampled responses and context directions fitted at the same layer.
2. **Replication** on a second model family.
3. **Gap metric:** combine the detector readings (probe, NLA) with the steering bounds into a per-model evaluation-awareness gap: how much context the detectors read, against how much of the behavioral difference that reading can account for.

## Artifacts
- `results/qwen2.5-7b-instruct/`: `steer_v3_pii_L10.jsonl` (now with `random_3..19`), `steer_v3_pii_L10.summary.json`, `steer_v3_pii_L10_sampled.jsonl`, `steer_v3_pii_L10_sampled.summary.json`
- `scripts/`: `steer.py` (`--samples`), `steer_analysis.py` (sample averaging, random-null rank), `judge.py` (skips rows already judged)
