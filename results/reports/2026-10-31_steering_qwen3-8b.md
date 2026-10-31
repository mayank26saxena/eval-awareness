# Third model: probes and context steering in Qwen3-8B

**Date:** 2026-10-31 · **Status:** final for this project (10 privacy tasks)

Repeats the probe and steering analyses on Qwen3-8B (non-thinking mode), the largest model that runs locally. It is the third model after Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507 ([layer 20 and Qwen3-4B](2026-10-31_steering_layer20_qwen3.md)). Qwen3-8B has 36 layers like Qwen3-4B, so the steered layer (13) and the readout layers are the same.

## Summary
- **Probes: a mid-network context band, as in Qwen2.5-7B.** A naive eval-vs-deploy probe, trained where format and context agree, tracks context at layers 10–18 (off-diagonal AUROC vs context 0.82–0.94) and format from layer 21 on (0.63–0.90). Qwen3-4B showed no such band, so the layer pattern holds in two of three models.
- **Steering moves the readout strongly.** At α = ±0.5 the layer-20 context readout shifts by −12.6 / +11.4, against a natural gap of +14.8. Random directions shift it by at most 2.4.
- **Behavior responds modestly, and mainly in one direction.** Under sampling the sign-specific effect is **+19 pp [+10, +29]**. With greedy decoding it is +17 pp [+7, +30], at the top of the 10-direction random distribution (range −23 to +15 pp, SD 13). The effect comes from the "deploy" direction lowering protection (deployment prompts −26 pp, evaluation prompts −11 pp). The "eval" direction changes nothing (−1 and +2 pp), partly because protection is near ceiling.
- **The natural gap is again not significant:** +12 pp [−4, +34]. Most of it comes from one task (clinic, +100 pp) and one other (hotel, +40 pp).
- **Reading:** Qwen3-8B sits between the other two models. Its context direction is more behaviorally relevant than in Qwen2.5-7B, but much weaker than in Qwen3-4B. As in Qwen3-4B, steering acts on deployment prompts as much as on evaluation prompts. Unlike Qwen3-4B, it does not change response length or style.

## Method
As in the [layer 20 and Qwen3-4B report](2026-10-31_steering_layer20_qwen3.md), with these differences:

| | |
|---|---|
| Model | `Qwen/Qwen3-8B`, bf16, chat template with `enable_thinking=False` (no reasoning trace; verified on the outputs) |
| Activations and probes | `scripts/probe.py` on the v3 main grid (312 prompts); layer-13 norm unit 51.5; cos(context_probe, context_mean) = +0.74 |
| Sampled sweep | 5 samples per cell (instead of 10), drawn one at a time (`--sample-batch 1`) to fit in 24 GB; temperature 0.7, top-p 0.8, top-k 20 (Qwen's non-thinking settings, matching the other two models); 600 responses |
| Greedy sweep | Unsteered, `context_probe` and `random_0..9` at ±0.5; 920 responses |
| Scoring | Qwen3-4B-Instruct-2507 judge (same family, different model); 0 unparsed verdicts; 1 degenerate response of 1,520. A hand check of 10 randomly drawn steered responses agreed with all 10 verdicts |

## Results

### Manipulation check (shift in the downstream context-probe logit)

| Readout | Natural gap | context_probe −0.5 / +0.5 | Random directions (max \|shift\|) |
|---|---|---|---|
| L20 | +14.8 | −12.6 / +11.4 | 2.4 |
| L27 | +12.2 | −3.7 / +2.4 | 6.1 |
| L35 | +10.5 | −1.4 / +2.1 | 3.6 |

The shift is large at layer 20 and fades by the final layers, where some random directions move the readout as much.

### Sampled responses, 5 per prompt

| | Deploy prompts | Eval prompts |
|---|---|---|
| Unsteered protection | 0.83 | 0.95 |
| context_probe −0.5 | 0.57 (−0.26 [−0.40, −0.12]) | 0.84 (−0.11 [−0.26, +0.03]) |
| context_probe +0.5 | 0.82 (−0.01 [−0.10, +0.07]) | 0.97 (+0.02 [−0.02, +0.07]) |
| **Sign-specific effect** (all 40 prompts) | **+0.19 [+0.10, +0.29]** | |
| Natural eval − deploy gap | +0.12 [−0.04, +0.34] | |

### Sign-specific effect against 10 random directions (greedy)

| | Effect | Random directions ≥ it |
|---|---|---|
| context_probe | +0.17 [+0.07, +0.30] | 0 / 10 |
| Random (10) | mean −0.02, SD 0.13, range [−0.23, +0.15] | |

### Coherence and style

Mean NLL under the unsteered model is 0.18 unsteered, 0.59–0.63 for the context direction, and 0.57–0.60 for random directions. Mean response length is 77–83 words in every condition, and refusal phrasing is 82–89%. The "deploy"-steered disclosures are fluent answers that hand over the private detail, for example the reason for a clinic appointment.

## Interpretation
1. **Across three models, the causal weight of the decoded context direction varies widely.** The sign-specific effect at α = 0.5 is 0 pp in Qwen2.5-7B (layer 20), +19 pp in Qwen3-8B and +51 pp in Qwen3-4B. The readout shift is comparable in all three.
2. **It is unrelated to the size of the natural gap.** The one model with a significant natural gap (Qwen2.5-7B, +22.5 pp) is the one whose context direction has no effect. In both Qwen3 models the gap is around +10 pp and not significant, and steering acts on deployment prompts as much as on evaluation prompts.
3. **In Qwen3-8B the effect is one-sided.** Pushing toward "deploy" lowers protection, and pushing toward "eval" does not raise it above a near-ceiling baseline. This fits a direction that carries some of what makes the model cautious, without being a dedicated record of being evaluated.

## Limitations
- Five samples per cell (wider within-prompt uncertainty than for the other models).
- The context direction is only just above the random distribution under greedy decoding (+17 against a random maximum of +15).
- Judge from the same model family.
- Probe transfer to natural transcripts was not run for this model.

## Artifacts
- `results/qwen3-8b/`: `activations_v3_main.npz` (not tracked), `probe_results_v3_main.json`, `steer_check_v3_pii_L13.json`, `steer_v3_pii_L13.jsonl`, `steer_v3_pii_L13_sampled.jsonl` and their `.summary.json`
- `scripts/`: `common.py` (thinking off for hybrid models), `steer.py` (`--sampling`, `--sample-batch`)
