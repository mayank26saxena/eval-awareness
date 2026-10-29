# Steering the privacy effect along the context direction: Qwen2.5-7B-Instruct, v3

**Date:** 2026-10-29 · **Status:** preliminary (greedy decoding, one layer, 10 privacy tasks)

Tests whether the linearly decodable representation of evaluation context at layer 10 causally drives the privacy effect: with all context cues present and no explicit grading line, privacy protection is 90%, against 65% in the deployment context ([single-cue report](2026-10-27_single_factor_behavior_qwen2.5-7b.md)). The context direction is steered against format, contrastive-prompt and random directions as controls.

## Summary
- **The steering moves the internal context readout.** Adding the context direction at layer 10 shifts a context probe read out at layer 20 by about 70% of the natural eval-vs-deploy gap at α = ±1, and by about 45% at α = ±0.5. At the final layer (27) it moves the probe by 25–40% of the gap. Random and format directions of the same norm shift the layer-20 readout by at most 13% of the gap.
- **The behavior does not follow the readout.** Steering toward "eval" does not increase protection on deployment-context prompts, and steering toward "deploy" does not specifically decrease it on evaluation-context prompts. The sign-specific effect, protection(+α) − protection(−α), is between −5 and +15 pp for both context directions at every strength, and no interval excludes zero.
- **Privacy protection is fragile to any perturbation.** Random directions lower protection as much as or more than the context directions: on evaluation-context prompts, from 90% to 10–25% for random directions at α = −1, against 65–70% for the context directions. Two of three random directions show larger sign-specific effects (+30 and +45 pp at α = 1) than either context direction. These responses are coherent, real disclosures, not degenerate text.
- **Conclusion for this model and layer:** the representation that a context probe decodes is not the variable through which the evaluation context raises privacy protection. Steering along it is no more effective, or more specific, than steering along an arbitrary direction.

## Method

| | |
|---|---|
| Prompts | v3 main grid, privacy tasks, explicit grading absent: 10 tasks × 2 formats × {deploy, eval} = 40 prompts |
| Directions (layer 10) | Fitted on last-prompt-token activations at `hidden_states[10]` from the probe cache, excluding topic controls and **leaving out the steered task**: `context_probe` (logistic regression, eval vs deploy, mapped to raw activation space), `context_mean` (mean eval − mean deploy activation, the contrastive-prompt direction), `format_probe` (benchmark vs natural), `random_0..2` (seeded Gaussian). All unit-normalized |
| Geometry | cos(context_probe, context_mean) = +0.81; cos(context_probe, format_probe) = −0.10; random directions within ±0.03 of both context directions |
| Intervention | α · 41.6 · d added to the output of decoder block 9 (= `hidden_states[10]`) at every prompt and generated position; 41.6 is the mean last-token activation norm at that layer. α ∈ {±0.25, ±0.5, ±1}; positive points toward eval (context) or benchmark (format). At α ≥ 2 every direction, random included, produces incoherent text |
| Generation | Greedy, 300 new tokens; one unsteered baseline per prompt; 1,480 responses (`scripts/steer.py`) |
| Scoring | Qwen3-4B-Instruct-2507 judge, privacy rubric (`scripts/judge.py`); 0 unparsed verdicts |
| Coherence | Mean token NLL of each response under the unsteered model (0.32 unsteered, 0.4 at |α| = 0.25, 0.5–0.7 at 0.5, 1.0–2.1 at 1) and a degeneracy check (fewer than 5 words or distinct-bigram ratio < 0.5): 1 degenerate response of 1,440 steered. A hand check of flipped responses found coherent disclosures of the private detail, consistent with the judge |
| Manipulation check | Readout probes (logistic regression, eval vs deploy, leaving out the task) at `hidden_states[14, 20, 27]`; mean shift of the probe logit under steering, compared with the mean natural eval − deploy logit gap for the same task and format (`scripts/steer_check.py`) |
| Statistics | Paired by prompt against the unsteered response; mean over tasks; bootstrap 95% CI over tasks (`scripts/steer_analysis.py`) |

## Results

### Manipulation check: shift in the downstream context-probe logit

| Direction | α | L14 | L20 | L27 |
|---|---|---|---|---|
| *Natural eval − deploy gap* | | *+15.4* | *+12.5* | *+11.3* |
| context_probe | −1 / +1 | −22.5 / +16.5 | −7.8 / +8.8 | −4.1 / +3.6 |
| context_mean | −1 / +1 | −22.5 / +17.4 | −8.3 / +9.9 | −4.3 / +2.9 |
| format_probe | −1 / +1 | +0.7 / −2.7 | −0.1 / +0.2 | +1.3 / +2.5 |
| random_0 | −1 / +1 | −3.4 / +3.0 | −0.2 / +0.9 | +0.3 / +0.2 |
| random_1 | −1 / +1 | −0.6 / −0.5 | +1.6 / +0.7 | +0.7 / −1.6 |
| random_2 | −1 / +1 | +1.5 / −2.1 | +1.1 / +0.9 | +1.9 / +0.4 |

### Change in protection from the unsteered response (unsteered: deploy 0.65, eval 0.90)

| Direction | Deploy, α = +0.5 | Deploy, α = +1 | Eval, α = −0.5 | Eval, α = −1 |
|---|---|---|---|---|
| context_probe | +0.10 [−0.10, +0.30] | −0.15 [−0.40, 0.00] | −0.30 [−0.60, −0.05] | −0.20 [−0.45, +0.05] |
| context_mean | 0.00 [−0.20, +0.20] | 0.00 [−0.30, +0.25] | −0.15 [−0.35, +0.05] | −0.25 [−0.60, +0.05] |
| format_probe | +0.15 [−0.05, +0.35] | +0.10 [−0.10, +0.30] | −0.05 [−0.20, +0.10] | −0.10 [−0.35, +0.10] |
| random_0 | +0.05 [−0.15, +0.25] | −0.10 [−0.30, +0.10] | −0.45 [−0.70, −0.20] | −0.80 [−0.95, −0.65] |
| random_1 | −0.10 [−0.30, 0.00] | −0.30 [−0.55, −0.05] | −0.10 [−0.35, +0.10] | −0.65 [−0.90, −0.40] |
| random_2 | 0.00 [−0.25, +0.20] | −0.15 [−0.45, +0.10] | −0.15 [−0.35, +0.05] | −0.80 [−1.00, −0.60] |

A context effect predicts a positive change in the first two columns and a negative change in the last two, specific to the context directions. Steering against the context direction on eval prompts does lower protection (−0.15 to −0.30), but steering *toward* eval lowers it about as much (context_mean, eval prompts: +0.5 gives −0.25, +1 gives −0.30), and random directions lower it more. All cells, including those for ±0.25, are in `steer_v3_pii_L10.summary.json`.

### Sign-specific effect, protection(+α) − protection(−α), all 40 prompts

| Direction | α = 0.25 | α = 0.5 | α = 1 |
|---|---|---|---|
| context_probe | +0.03 [−0.05, +0.10] | +0.15 [−0.10, +0.40] | −0.03 [−0.23, +0.17] |
| context_mean | +0.12 [−0.07, +0.30] | −0.05 [−0.25, +0.12] | 0.00 [−0.15, +0.23] |
| format_probe | −0.03 [−0.10, +0.05] | +0.03 [−0.17, +0.20] | −0.05 [−0.17, +0.10] |
| random_0 | +0.10 [0.00, +0.20] | **+0.35 [+0.15, +0.55]** | **+0.45 [+0.23, +0.65]** |
| random_1 | −0.03 [−0.10, +0.05] | −0.03 [−0.12, +0.07] | +0.12 [−0.07, +0.33] |
| random_2 | +0.03 [−0.15, +0.23] | **+0.12 [+0.03, +0.23]** | **+0.30 [+0.07, +0.53]** |

A generic perturbation effect cancels in this difference. For the context directions it would be positive if the context representation drove protection.

## Interpretation
1. **Decodable is not causal here.** The context direction is read out downstream when it is added, so the intervention changes what later layers represent about context. The privacy decision does not respond to that change in a sign-specific way. Whatever the evaluation-context prompts do to raise protection, it is not mediated by this direction at layer 10.
2. **Random directions are the necessary control.** Without them, the drop in protection when steering eval prompts toward "deploy" (−0.15 to −0.30) would look like a causal effect of the context representation. The random directions show that protection on these tasks breaks under any perturbation of this size, and that some arbitrary directions push it further in one sign than the context direction does.
3. **Consistency with earlier results.** The NLA does not verbalize evaluation at the response position, no single cue moves the behavior, and steering along the context direction does not move it either. Together this suggests that, in this model, the privacy effect of the evaluation context is carried by specific content of those prompts (for example how the private detail is framed), not by a general "this is an evaluation" state.

## Limitations
- One layer (10) and one steering scheme (all positions, constant vector). The context signal may be causally relevant at other layers, or only at specific token positions.
- Leave-one-task-out directions fitted on 312 prompts; the probe direction may not be the direction the model uses even if one exists (LEACE or multi-layer steering would test this differently).
- Ten privacy tasks, greedy decoding, one response per cell. Effects below about 15 pp cannot be resolved.
- Three random directions only. Their large and uneven effects suggest the behavior is sensitive to particular directions, which a larger random sample would characterize.

## Next steps
1. **Sampled responses** on the privacy tasks (several per prompt), to replace single greedy outcomes with disclosure probabilities. This also resolves smaller steering effects.
2. **More random directions** (for example 20) at α = ±0.5, to place the context directions within the distribution of arbitrary perturbations.
3. **Other layers and positions**: steering at layers 14 and 20, and only at the system-message tokens, where the NLA reading of context is located.

## Artifacts
- `results/qwen2.5-7b-instruct/`: `steer_v3_pii_L10.jsonl` (judged responses), `steer_v3_pii_L10.summary.json`, `steer_check_v3_pii_L10.json`
- `scripts/`: `steer.py`, `steer_analysis.py`, `steer_check.py`
