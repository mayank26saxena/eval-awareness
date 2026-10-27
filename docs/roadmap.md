# Roadmap

Status as of 2026-10-27. Background and hypotheses are in [`research_statement.md`](research_statement.md).

## Phase overview

| Phase | Scope | Status |
|---|---|---|
| 0. Pipeline | Dataset schema, rendering, validation, generation, scoring, probing | ✅ Done |
| 1. Pilot | Qwen2.5-0.5B smoke test; Qwen2.5-7B behavior and probes | ✅ Done ([report](../results/reports/2026-10-25_pilot_qwen2.5-7b.md)) |
| 2. Dataset v2/v3 | Fix the bare-shell stake confound (v2); scale to 42 tasks (v3) | ✅ Done |
| 3. Scoring | LLM judge calibrated against hand labels | ✅ Local judge, 98% agreement; a stronger independent judge is planned |
| 4. Probes and transfer | Layer-wise format/context probes; transfer to natural eval/deploy transcripts; second model family | ✅ Done for Qwen2.5-7B and Qwen3-4B ([report](../results/reports/2026-10-26_round2_qwen.md)) |
| 5. Activation readers | NLA (Qwen2.5-7B L20, Gemma-3-12B L32, Gemma-3-27B L41), Activation Oracle | In progress: Qwen2.5-7B pre-response ([report](../results/reports/2026-10-26_nla_qwen2.5-7b.md)) position sweep ([report](../results/reports/2026-10-27_nla_positions_qwen2.5-7b.md)) and single-factor artificiality analysis ([report](../results/reports/2026-10-27_nla_artificiality_qwen2.5-7b.md)) done |
| 6. Causality | LEACE format-free direction, NLA-edit steering, contrastive and random controls | Planned |
| 7. Scale-up and write-up | Gemma-3-27B on GPU; workshop paper | Planned |

## Phase details

### 2. Dataset
- [x] v1: 6 behavior tasks plus 2 topic controls, EvalAwareBench F1–F8 taxonomy
- [x] v2: bare (F4) shells keep the user's stake
- [x] v3: 28 honesty, 10 privacy and 4 topic-control tasks (42 total)
- [ ] Realism-edit arm, using the published realism-rewrite prompt on benchmark-format items
- [ ] Subtler, less lexical context cues, since probe AUROC currently saturates
- [ ] Optional external test set: EvalAwareBench tasks

### 3. Scoring
- [x] Keyword heuristic (baseline only: it disagreed with hand labels on 6 of 48)
- [x] Local LLM judge (Qwen3-4B-Instruct-2507): 47/48 agreement with hand labels
- [ ] Independent judge from a different model family, calibrated on the same hand labels

### 4. Probes and transfer
- [x] Layer-wise context, format and explicit probes; naive probe trained on the diagonal and tested off-diagonal
- [x] Transfer to natural transcripts (aware-bench), with length and language controls
- [x] Replication on a second family (Qwen3-4B-Instruct-2507): the layer pattern does not replicate
- [ ] Generalization to subtler cues

### 5. Activation readers
- [x] NLA explanations at Qwen2.5-7B layer 20 (pre-response token), scored with the published grader: no eval-awareness readings
- [x] NLA position sweep (system end, user end, pre-response, 8 response tokens; any-of-N): no evaluation readings beyond explicit grading text; context read as artificiality at the system message
- [ ] Replicate the published manipulations (eval vs deploy, contrastive steering, realism edits) on open models
- [x] Validated measures separating artificiality from evaluation framing; single-factor attribution: contrived details (F2) drive the artificiality reading, and only explicit grading text produces evaluation language
- [ ] Multiple NLA samples per activation, to resolve single-factor effects below ~10 pp
- [ ] AR grounding test: ablate eval-awareness claims and measure the change in reconstruction error
- [ ] Activation Oracle answers, averaged over question paraphrases, with known-answer controls

### 6. Causality
- [ ] Format-free direction via LEACE
- [ ] Steering with format-free, NLA-edit, contrastive-prompt, format and random directions; single and multi-layer
- [ ] Activation patching between context conditions
- [ ] Capability and coherence controls
- [ ] Optional: neuron-level analysis. Rank MLP neurons by the alignment of their output weights (`W_down` columns) with the context and format probe directions, then steer the top candidates and record rollouts. Method after Lyra C's neuron-explorer (steering-sweep characterization of individual MLP neurons in Qwen3.5-4B).

### 7. Scale-up and write-up
- [ ] Gemma-3-12B / 27B runs on an 80 GB GPU (see [`compute.md`](compute.md))
- [ ] Optional: Llama-3.3-70B via NDIF
- [ ] Main figure: behavior gap · detector decomposition · steering dose-response
- [ ] Workshop paper or technical report; dataset and harness release

## Decision gates
- **Behavior gap.** If the eval/deploy behavior gap is under ~5 percentage points at the target scale, add behaviors with a known larger gap (real-vs-hypothetical pairs, sandbagging) before running the causal experiments.
- **Detector validity.** If every activation reader follows format only, report that as the primary result and reduce the scope of Phase 6.

## Risks

| Risk | Mitigation |
|---|---|
| Concurrent publication of an NLA/AO validation | Release intermediate results early; the causal and grounding analyses remain distinct |
| Weak eval awareness at small scale | Main results on ≥27B models; smaller models used for pipeline development only |
| Lexically obvious cues saturate probes | Subtler cue variants; transfer to natural transcripts |
| Judge bias | Calibration against hand labels; a second judge from a different family |
| Nondeterministic generation | Batch size 1 with greedy decoding; multiple seeds for sampled runs |
