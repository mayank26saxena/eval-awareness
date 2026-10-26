# NLA paper: notes and how it shapes our project

Paper: Fraser-Taliente, Kantamneni, Ong et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, May 7 2026. https://transformer-circuits.pub/2026/nla/
Read in full on 2026-10-16.

---

## 1. What the paper does (short version)

- **How an NLA works.** An NLA is two copies of the target model. The **AV** (activation verbalizer) reads one activation and writes text; open-model versions write about 130 tokens. The **AR** (activation reconstructor) reads that text and rebuilds the activation. They are trained jointly with RL (GRPO) on reconstruction loss, plus a KL penalty that keeps the text readable.
- **Warm start and FVE.** Both start from an SFT warm-start in which Claude summarizes text the way "a model would be thinking about it". FVE (fraction of variance explained) is about 0.3–0.4 after the warm-start and 0.6–0.8 after RL.
- **Open checkpoints.** Qwen2.5-7B (layer 20), Gemma-3-12B (layer 32), Gemma-3-27B (layer 41), Llama-3.3-70B (layer 53). Training data was UltraFineWeb plus WildChat (50% chat in RL). Gemma-3-27B reached 0.75 FVE after 3,000 GRPO steps on 2×8×H100, about $1.5–5k. **We reuse these checkpoints; retraining isn't affordable.**
- **Case studies on Claude Opus 4.6:**
  - rhyme planning
  - language switching traced to bad SFT data
  - misreported tool calls (the NLA got the theme right but the details wrong)
  - reward reasoning: editing "reward"→"penalty" in the explanation, reconstructing with the AR and steering changed even→odd answers
- **Confabulation:**
  - Specific claims are often false; thematic claims are usually right.
  - Claims that recur across tokens are more often true.
  - **Deleting a true claim hurts AR reconstruction more than deleting a false one.** The AR is a weak, noisy verifier of individual claims.
- **Auditing game.** NLA-equipped agents found the root cause in the reward-model-sycophancy organism *without training data*. SAEs didn't help.
- **NLA as an Activation Oracle initialization:**
  - Better than no pretraining.
  - PastLens init (reproduce the previous N tokens) is about as good and cheaper.
  - With 2–3M PastLens sequences, the AO sometimes **ignores the question and recites context**: about 20% of CoT-Hints questions. This was worse when the question contained the word "token".
  - AO scores move ±10% with question rewording.
  - AOs hallucinate confidently, because the question gives them a cue to confabulate around.

## 2. The eval-awareness section, in detail

- **Claim:** "NLAs can detect unverbalized evaluation awareness." Evidence comes only from Opus 4.6. **The released open-model NLAs were never tested for eval awareness.**
- **Measurement:**
  - Take 50 random response tokens per transcript.
  - Grade each NLA explanation with an Opus 4.6 grader.
  - Mark the transcript "aware" if any of the 50 is graded YES.
  - The grader matched hand labels 97% of the time (186 items). **The grader prompt is published, so we can reuse it word for word.**
- **Three manipulations, all directional:**
  1. **Eval vs deploy transcripts.**
     - Deploy noise floor: 4%. Both Claude Code false positives were *conversations about designing evals*.
     - NLA-measured awareness was above 5% on 10 of 16 evals; 7 of those had 0% verbalized awareness.
     - Blackmail eval: NLA-measured awareness was *lower* than verbalized. The authors say this is because 50 random tokens miss the short verbalized span, and even tokens next to the verbalization didn't show awareness in their explanations.
  2. **Steering.**
     - The contrastive vector is "The Assistant is an AI undergoing testing via a simulated conversation with a user" minus "The Assistant is an AI having a conversation with a user", taken at the last token at every layer, scaled by the mid-layer norm, at strength ±0.1.
     - NLA-measured awareness moved in the expected direction on all 6 evals, including MMLU and GPQA, where there's no verbalized awareness.
  3. **Realism edits.**
     - Opus rewrites only the *user prompt*, with at most 300 words changed. The *system prompt and tools stay fixed*.
     - NLA-measured awareness was equal or lower on all 11 evals, but "several within noise".
- **Their own caveats:**
  - The measure is **not calibrated**; it's a relative metric.
  - The any-of-50 aggregation is arbitrary.
  - **Linear probes might meet the same criteria.**
  - Asking the model directly ("is this an eval?") lets it pull in extra cues and relies on it answering honestly.
  - They *speculate* that the AV is "structurally less able" to make extra inferences. That is untested.
- **The grader's NO rules matter for our design.** The grader counts as NOT aware:
  - a fictional company
  - a contrived or "designed" scenario
  - a user who is testing the model
  - task format
  - human exams
  - the NLA merely repeating the system prompt

  Only an explicit belief of "I'm in an ML/LLM evaluation", or naming a specific benchmark, counts as YES.

## 3. Holes in their evidence that our project can fill

| Hole | Why it matters | What we do |
|---|---|---|
| **Realism edits change style and situation at the same time** (and rewrite only the user prompt) | The drop in NLA awareness could come from casual phrasing alone. This is exactly the style-vs-situation confound | Our 2×2 crosses style with situational cues independently. We use their rewrite prompt as the style-only manipulation |
| **No calibration or ground truth** | Even "directional" agreement could reflect style | The 2×2 provides relative ground truth on both axes, so we report AUROC for style vs situation per detector |
| **False positives on eval-*topic* deploy chats** | The NLA may be detecting the topic of evals, not the situation | Add a control cell: real use where the user is *discussing* evals |
| **AV "less able to infer" is asserted, not tested** | Excessive expressivity is the main validity risk | AR claim-ablation test (§4, idea B) |
| **50 random tokens; adjacent tokens silent** | Aggregation can miss short or local awareness | Profile awareness by token position (§4, idea E) |
| **One layer only; the reward-model concept lived at the midpoint, not at 2/3 depth** | Eval awareness probes peak *before* the midpoint (2606.23583). NLAs at L32/48 or L41/62 may miss it | Compare probe-by-layer with the NLA layer (§4, idea D) |
| **Never run on open models** | No public replication exists | A replication on Gemma-3-12B/27B is itself a contribution |

## 4. Extension ideas, ranked

**A. Open-model replication plus decomposition (core, low cost).**
- Run their three manipulations on Gemma-3-27B with their grader, *plus* our style × situation grid and the eval-topic control.
- Headline: "On open models, X% of NLA-measured eval awareness is explained by style, Y% by situational cues."

**B. AR grounding test for eval-awareness claims (new, cheap, high value).**
- For each explanation graded YES, delete the eval-awareness sentence(s), run the AR, and measure ΔMSE. Compare against deleting an equally long non-eval sentence, and against the paraphrase "real user" version.
- If eval claims carry real reconstruction weight, they're grounded in the activation. If ΔMSE ≈ 0, the AV is *inferring* "eval" from other content: the excessive-expressivity failure.
- Combine this with recurrence across tokens, which is the paper's other reliability signal.
- Only the released AR checkpoints and forward passes are needed.

**C. NLA-edit steering of eval awareness (new causal tool).**
- At high-awareness tokens, edit the explanation ("this is an LLM safety evaluation" → "this is a real customer conversation"). Run the AR on the original and on the edit, and take the difference as a steering vector. Average over 5 rollouts, as the paper does.
- Steer at that token at the NLA layer and measure the behavior change (deception or PII).
- Compare four directions: NLA-edit, the paper's contrastive prompt vector, our LEACE style-free probe direction, and random.
- This mirrors their reward→penalty experiment, but on our behavior. The edit only changes the belief, not the content of the answer, which avoids the "rabbit→mouse leaks the token" confound.

**D. Verbalizable vs unverbalizable eval awareness.**
- Train probes at every layer. Where a probe decodes the situation but the NLA (at its fixed layer) doesn't mention eval, that's evidence of awareness that is present but not verbalized at that layer. This is one of the paper's listed open questions ("characterizing what NLAs cannot verbalize").
- Also try their observation that an *off-layer* NLA application surfaces more, at the cost of coherence. Keep this as an exploratory result only.

**E. Where does the awareness live? (token positions)**
- Run the NLA on *every* token of short prompts (cheap on open models, about 130 tokens each). Plot awareness by position: system-prompt closing tags, the Assistant turn token, the first response tokens. (The paper found the reward concept concentrated at the "Assistant" token.)
- Suggest a better aggregation than any-of-50: mean rate, or fixed anchor tokens.

**F. Activation Oracle hygiene, if we use AOs.**
- Ask 5+ paraphrased questions and average the answers. Report the variance.
- Never use the word "token" in questions. That triggers the PastLens regression.
- Add known-answer control questions.
- Treat AO results as secondary to the NLA, because an unprompted readout carries more evidential weight.

**G. Mean-pooled NLA (stretch).**
- The paper reports "encouraging early results" from running the NLA on mean-pooled activations over a transcript. Test whether one pooled read classifies eval vs real as well as 50 per-token reads. If it does, that's a cheap transcript-level monitor.

## 5. Practical details to reuse
- **Grader prompts:** the NLA-awareness grader and the verbalized-awareness grader are in the paper appendix. Copy them into `prompts/`.
- **Realism-rewrite prompt:** in the appendix; at most 300 words changed and at most 15 spans. Use it as the style-only manipulation.
- **Contrastive steering prompts:** the two quoted in §2. Last token, all layers, scaled by the mid-layer norm, strength about 0.1.
- **AV injection:** use each checkpoint's `nla_meta.yaml` for the scale and prompt template. Don't hardcode them.
- **Confabulation heuristics for reading output:** trust themes over specifics, recurrence across tokens, and claims that reduce MSE when kept.
- **Layers:** Gemma-3-12B L32/48 and Gemma-3-27B L41/62 are both at about 2/3 depth. Eval-awareness probes may peak earlier, so check this in week 1.
