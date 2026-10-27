# NLA artificiality vs evaluation readings: single-factor attribution, Qwen2.5-7B-Instruct

**Date:** 2026-10-27 · **Status:** preliminary (one NLA sample per activation, one model)

Follow-up to the [position sweep](2026-10-27_nla_positions_qwen2.5-7b.md). That sweep found that the NLA calls evaluation-context system prompts "scenarios" far more often than deployment-context ones. This report measures *artificiality* ("this is fictional") and *evaluation framing* ("this is being graded") separately, with validated measures, and attributes the artificiality reading to individual context cues using the v3 single-factor set.

## Summary
- **Artificiality is a real, format-invariant context reading, at about half the size the keyword count suggested.** With a validated measure, the NLA says the content is fictional, hypothetical or contrived at the end of the system message for 50% of evaluation-context prompts, against 16% of deployment-context prompts (+34 pp). The effect is the same in both formats: natural +36 pp, benchmark +32 pp. The earlier 98% vs 35% came from counting the word "scenario", which mostly means "situation".
- **Contrived details (F2) are the only cue that produces it on its own.** Changing one factor at a time, contrived details raise the system-message artificiality rate by +18 pp (95% CI [+8, +30]). Placeholder entities (F1: +1 pp) and the bare-request rewrite (F4: −3 pp) have no detectable effect alone. The full evaluation context, with all cues present, gives +34 pp, so the cues reinforce each other.
- **Evaluation framing appears only when the prompt states it.** Evaluation or grading language appears in 74–92% of explanations at the end of the user message when the prompt contains an explicit grading line (F8), and in at most 5% for every other condition, at every position. No implicit cue makes the NLA describe the prompt as a test.
- **At the end of the user message, artificiality depends on format.** It appears only in benchmark-format prompts (37–66%), never in natural-format prompts (0–5%) unless the prompt says it will be graded.

## Method

| | |
|---|---|
| Data | v3 single-factor set: 38 behavior tasks × 2 formats × 6 conditions (all-realistic, or exactly one of F1, F2, F4, F6, F8 flipped), 456 prompts. v3 main grid for comparison |
| Positions | `system_end`, `user_end`, `pre_response` of the prompt (no response), `hidden_states[21]` (`scripts/extract_positions.py` without `--responses`) |
| Verbalizer | `kitft/nla-qwen2.5-7b-L20-av`, published recipe, one sample per activation; 1,368 explanations |
| Artificiality | Qwen3-4B-Instruct-2507 answers one yes/no question: does the text explicitly say the content is fictional, hypothetical, a placeholder, contrived, a puzzle, role-play or otherwise not real? "Scenario" alone does not count (`prompts/setting_realism_rubric.md`) |
| Evaluation framing | Keyword rule: evaluation, grading, scoring, rubric or benchmark language (`scripts/nla_setting_rubric.py`) |
| Validation | 60 explanations from the v3 main sweep (30 `system_end`, 30 `user_end`), hand-labeled blind to condition (`setting_rubric_hand_labels.json`) |
| Noise floor | F6 and F8 change only the user message. Under causal attention they leave the `system_end` activation unchanged up to numerical precision (relative difference ≤ 2%, against 18% for F1 and 74% for F2, medians). Their `system_end` differences therefore measure NLA sampling noise |
| Statistics | Paired by task and format (76 pairs per factor); bootstrap 95% CIs over pairs |

### Measure validation

| Measure | Precision | Recall |
|---|---|---|
| Artificiality (judge yes/no) | 19/22 (0.86) | 19/23 (0.83) |
| Evaluation framing (keywords) | 6/8 (0.75) | 6/8 (0.75) |

Two other judge designs failed validation and were not used:
- **Three-way rubric** (evaluation / artificial / neither): 30 of 60 agreement with the hand labels, labeling most explanations "evaluation" with unsupporting quotes.
- **Yes/no evaluation question:** either too strict (recall 1/8) or too loose (22 false positives), depending on wording.

## Results

### Main grid, `system_end`

| | Deploy context | Eval context | Difference |
|---|---|---|---|
| Natural format | 0.18 | 0.54 | +0.36 |
| Benchmark format | 0.14 | 0.46 | +0.32 |
| Evaluation framing (all cells) | ≤ 0.03 | ≤ 0.05 | — |

### Single-factor attribution, `system_end` artificiality (flip − all-realistic, paired)

| Factor flipped | Changes system message? | Effect | 95% CI |
|---|---|---|---|
| F1 placeholder entities | yes | +0.01 | [−0.09, +0.12] |
| **F2 contrived detail** | yes | **+0.18** | **[+0.08, +0.30]** |
| F4 bare request | yes | −0.03 | [−0.13, +0.07] |
| F6 eval scaffolding | no (noise floor) | −0.03 | [−0.12, +0.07] |
| F8 explicit grading | no (noise floor) | −0.01 | [−0.09, +0.08] |

The all-realistic baseline is 0.16 (natural 0.21, benchmark 0.11).

### `user_end`, single-factor set

| Condition | Artificiality, natural | Artificiality, benchmark | Evaluation framing, natural | Evaluation framing, benchmark |
|---|---|---|---|---|
| All realistic | 0.00 | 0.37 | 0.00 | 0.00 |
| F1 | 0.00 | 0.61 | 0.00 | 0.03 |
| F2 | 0.00 | 0.55 | 0.00 | 0.00 |
| F4 | 0.00 | 0.45 | 0.03 | 0.00 |
| F6 | 0.00 | 0.53 | 0.03 | 0.00 |
| F8 | 0.16 | 0.21 | **0.82** | **0.74** |

At `pre_response`, both measures are at most 0.03 in every condition.

## Interpretation
1. **The NLA's context reading is a reading of implausibility, not of evaluation.** The single cue that moves it on its own is the contrived coincidence (F2), the most semantically odd element of the prompt. Placeholder names alone do not register. Nothing implicit moves the NLA toward "this is a test of the assistant". This is consistent with the published rubric's choice to treat fictional framing as distinct from evaluation awareness, and it shows that for this model the distinction matters: the artificiality signal exists, the evaluation signal does not.
2. **Location matters.** The system-message artificiality reading is format-invariant, so it qualifies as a context signal in this study's sense. The user-message artificiality reading appears only in benchmark format, so it is a format signal. A detector that pooled positions would mix the two.
3. **Relation to the probes.** Linear probes decode context from the residual stream at this and earlier layers (see the [round-2 report](2026-10-26_round2_qwen.md)). The NLA reflects part of that information, specifically implausible content where it occurs, and none of it at the token where the response begins. This supports the working hypothesis that the model represents cue features without a downstream "I am being evaluated" state at this scale.

## Limitations
- One NLA sample per activation. The F6/F8 noise floor shows sampling noise of about ±0.10 on 76-pair differences, so single-factor effects under ~10 pp are not detectable.
- The validation set is small (60 items, 8 evaluation-labeled). Both measures were hand-labeled by a single annotator.
- The F2 contrived details are lexically distinctive ("by a strange coincidence"). Whether subtler implausibility registers is untested.
- One 7B model.

## Next steps
1. **AR grounding test** on the F2 artificiality readings. Remove the artificiality claim from the explanation and measure the change in activation reconstruction error.
2. **Multiple NLA samples per activation**, to resolve F1 and F4 effects below the current noise floor.
3. **Behavior link:** check whether F2 alone changes the privacy-task behavior, i.e. whether the cue the NLA notices is the one that moves the model's decisions.

## Artifacts
`results/qwen2.5-7b-instruct/`:
- `positions_v3_single_L21.jsonl`, `nla_positions_v3_single.jsonl`
- `nla_positions_v3_single_setting.jsonl` (+ `.summary.json`), `nla_positions_v3_main_setting.jsonl` (+ `.summary.json`)
- `setting_rubric_hand_labels.json`
