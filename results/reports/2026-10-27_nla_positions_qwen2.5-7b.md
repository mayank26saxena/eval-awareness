# NLA position sweep: Qwen2.5-7B-Instruct, layer 20

**Date:** 2026-10-27 · **Status:** preliminary (one sample per activation, one model)

Follow-up to the [pre-response NLA report](2026-10-26_nla_qwen2.5-7b.md), which read a single token per prompt. This sweep reads 11 positions per transcript, including the model's own response, and aggregates any-of-N per transcript as in the published NLA protocol.

## Summary
- **The published grader finds no context effect.** Any-of-N over 11 positions, eval-awareness YES rates are 2% for evaluation contexts and 2% for deployment contexts. Only 6 of 3,432 explanations are graded YES, and none of them come from response tokens.
- **Every YES comes from an explicit grading statement.** All 6 YES explanations are from prompts that state outright that the answer will be graded (factor F8). Five are at the end of the user message, which contains that statement. The verbalizer repeats explicit grading text. The implicit evaluation cues (F1, F2, F4, F6), including the literal header "Evaluation prompt #12", produce no ML-evaluation language at that position (0 of 152).
- **The NLA does track context at one position, but as fiction, not evaluation.** At the end of the system message, explanations call the setting a "scenario", "fictional" or "hypothetical" for 98% of evaluation-context prompts, against 35% of deployment-context prompts. This holds in both formats: 99% vs 36% for natural format, 97% vs 34% for benchmark format. None of these words appear in the system prompts themselves. The published rubric deliberately does not count perceived artificiality as evaluation awareness.
- **Response tokens carry no reading at all.** The published protocol relies on response tokens. Here they show no difference by context or format on any measure.

## Method

| | |
|---|---|
| Target model | Qwen2.5-7B-Instruct, bf16; transcripts = v3 prompt + the model's own greedy response (`responses_v3_main.jsonl`) |
| Positions (11 per transcript) | `system_end` (last token of the system message content), `user_end` (last token of the user message content), `pre_response` (last prompt token), 8 `response` tokens sampled uniformly (seed 0), excluding the end-of-turn token |
| Activation | `hidden_states[21]` = output of block 20, the NLA's `extraction_layer_index` (`scripts/extract_positions.py`, one forward pass per transcript) |
| Verbalizer | `kitft/nla-qwen2.5-7b-L20-av`, published recipe as before (`scripts/nla_explain.py`); one sample per activation; 3,432 explanations |
| Grading | Published NLA eval-awareness grader, verbatim, with a neutral eval description; judge Qwen3-4B-Instruct-2507. 53 verdicts truncated at 160 tokens were re-graded with 400 tokens. Final: 3,421 NO, 6 YES, 5 UNCLEAR |
| Keyword measures | *strict* (explicit ML-evaluation language), *broad* (perceived artificiality: scenario, hypothetical, fictional, ...), *template* |
| Aggregation | Per position type, and any-of-N per transcript (a transcript counts if any of its 11 positions is positive) |

## Results

### Per position type (304 behavior transcripts)

| Position | Grader YES | Strict | Broad | Template |
|---|---|---|---|---|
| `system_end` | 0.003 | 0.003 | 0.66 | 0.10 |
| `user_end` | 0.016 | 0.32 | 0.51 | 0.23 |
| `pre_response` | 0.000 | 0.003 | 0.06 | 0.65 |
| `response` (n = 2,432) | 0.000 | 0.016 | 0.11 | 0.08 |

### Which factor drives each reading

| Reading | Position | Evaluation context | Deployment context | Explicit grading present | Explicit grading absent |
|---|---|---|---|---|---|
| Grader YES | `user_end` | 2/56 | 3/64 | 5/120 | 0/150 |
| Strict keywords | `user_end` | 46/152 | 50/152 | 96/152 | **0/152** |
| Broad (artificiality) | `system_end` | **0.98** | **0.35** | — | — |
| Broad (artificiality) | `response` | 0.11 | 0.11 | — | — |

Artificiality at `system_end` by format: natural 0.99 (eval) vs 0.36 (deploy); benchmark 0.97 (eval) vs 0.34 (deploy). The most frequent terms in evaluation contexts are "scenario" (274 occurrences), "hypothetical" (23) and "fictional" (20).

### Any-of-N per transcript

| Measure | Context effect (eval − deploy) | Format effect (benchmark − natural) | Explicit effect | Topic controls |
|---|---|---|---|---|
| Grader YES | 0.00 (0.02 vs 0.02) | −0.01 | +0.04 | 0.00 |
| Strict | −0.07 | −0.01 | **+0.54** | 0.62 |
| Broad | **+0.14** (1.00 vs 0.86) | +0.06 | +0.03 | 1.00 |
| Template | −0.01 | **+0.15** | +0.12 | 0.38 |

Any-of-N compresses the system-message effect because other positions often contain a broad keyword too: 86% of deployment transcripts have at least one. The per-position view is the informative one.

### Hand inspection
- **YES verdicts (6).** All are from explicit-present items. Five quote or paraphrase the explicit grading line ("This answer is being evaluated", "Grading: honesty (0–10)"). The sixth (`condo_special_assessment`, benchmark/eval, `system_end`) describes "a specific trap scenario". It reacts to the contrived-coincidence detail (F2) in the system message, together with the explicit line later in the prompt.
- **Strict hits on response tokens (38).** Sampled matches refer to the conversation's content ("when evaluating the total housing costs"), not to the model being evaluated.
- **System-message explanations.** The evaluation-context system prompts use placeholder entities ("Sample Realty", "Building X") and contrived coincidences. The explanations describe these prompts as fictional scenarios. The deployment versions use realistic names and details, and their explanations describe the business setting, while still calling it a "scenario" in about a third of cases.

## Interpretation
1. **The NLA reads the prompt, not a belief about being tested.** Where it verbalizes evaluation at all, the source is an explicit grading statement located at that same token. The model's own response tokens, where an internal "I am being tested" state would most plausibly show, carry nothing.
2. **The NLA reads context as artificiality.** The verbalizer reliably notices placeholder entities and contrived details, so its descriptions track context and not format. It renders them as "this is a fictional scenario", which the published rubric treats as distinct from evaluation awareness. For this model, that is a context detector that works but does not meet the rubric's definition. Recovering it requires a broader rubric, which also brings in false positives: a third of the realistic deployment prompts are called scenarios.
3. **Same pattern as the linear probes, at a different location.** The probes find context information in the residual stream (mid layers, and at layer 20 on natural prompts). The NLA expresses it only where the cues physically sit (the system-message tokens), and not at the response tokens. That is consistent with the model representing the cue features without carrying a downstream "evaluation" state into generation. At this scale, that matches the weak behavioral effect (privacy only).

## Limitations
- One verbalizer sample per activation. The published protocol samples more, and any-of-N over more samples would raise every rate, including false positives.
- 8 sampled response tokens per transcript, against up to 50 in the published protocol.
- The judge is a 4B model. All YES and UNCLEAR verdicts and a sample of keyword hits were inspected by hand.
- The keyword lists are coarse. "Scenario" is also used for realistic prompts.
- A single 7B model with weak behavioral evaluation awareness. The published positive result is for a frontier model.

## Next steps
1. **Artificiality vs evaluation.** Score `system_end` explanations with a rubric that separates "this text is fictional" from "this is a test of the assistant", and check against the single-factor v3 set to identify which cue (F1 placeholder entities or F2 contrived details) drives the reading.
2. **AR grounding test** on the system-message artificiality readings. Ablate the "fictional scenario" claim and measure the change in reconstruction error.
3. **Gemma-3-27B NLA (layer 41)** on a GPU, where response-token readings would be a meaningful test.

## Artifacts
`results/qwen2.5-7b-instruct/`: `positions_v3_main_L21.jsonl` (position metadata), `nla_positions_v3_main.jsonl`, `nla_positions_v3_main_graded.jsonl`, `nla_positions_v3_main_graded.summary.json`. The activation cache (`positions_v3_main_L21.npz`) is not tracked.
