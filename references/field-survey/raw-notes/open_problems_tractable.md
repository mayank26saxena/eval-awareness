# Open problems in LLM interpretability that a low-compute independent researcher can make progress on (as of Oct 2026)

Research date: 2026-10-15. About 18 tool calls. Primary sources were favoured where they could be fetched. Compute estimates are my own inferences and are marked as such. Anything I could not source appears under Gaps.

## Q1. Which open problems do the agenda documents list explicitly?

### Takeaway
The most concrete and recent agenda is the Google DeepMind (GDM) mech-interp team's pair of posts from Dec 2025: "A Pragmatic Vision for Interpretability" and "How Can Interpretability Researchers Help AGI Go Well?". They name specific open questions in these areas:
- reasoning-model interpretability
- model diffing
- steering what fine-tuning learns
- monitoring with probes
- eval awareness
- model organisms
- latent reasoning

Sharkey et al. (Jan 2025, TMLR 2025) is still the umbrella taxonomy (methods / applications / socio-technical). Anthropic's 2025–26 Circuits posts point to open fronts (transcoder unfaithfulness, interference weights, attention, introspection, natural-language activation explanations), but none of their index summaries has an explicit "open questions" list. Neel Nanda's MATS stream pages give directions, not fixed project lists.

### Cited Findings
- **Sharkey et al., "Open Problems in Mechanistic Interpretability"** (arXiv 2501.16496, submitted 27 Jan 2025; published in TMLR 2025). Groups open problems into three areas: (1) methods need "conceptual and practical improvements", (2) how to apply methods to specific goals, (3) socio-technical challenges. Written by Apollo Research on behalf of Schmidt Sciences, with about 29 authors. — [arXiv](https://arxiv.org/abs/2501.16496); [TMLR listing](https://mlanthology.org/tmlr/2025/sharkey2025tmlr-open); [LessWrong post](https://www.lesswrong.com/posts/fqDzevPyw3GGaF5o9/paper-open-problems-in-mechanistic-interpretability)
- **Nanda, "A Pragmatic Vision for Interpretability"** (3 Dec 2025, GDM mech interp team). Advocates problems on the critical path to AGI safety, chosen by comparative advantage and measured with proxy tasks. Example proxy tasks:
  - interpreting the hidden goal of a model organism
  - stopping emergent misalignment without changing training data
  - predicting which prompt changes will stop an undesired behaviour

  Recommends trying prompting, steering, probing and reading CoT first. Says most existing techniques "struggle" on today's important behaviours (large models, agentic settings, long CoTs). — [post mirror](https://forum.nunosempere.com/posts/Tm2vqRZ6YMpiftCs4/a-pragmatic-vision-for-interpretability)
- **Companion post, "How Can Interpretability Researchers Help AGI Go Well?"** (Nanda et al., 1 Dec 2025) — [LessWrong](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well). Open questions it names:
  - **Reasoning models:** calls the area "extremely neglected". "How do we reverse-engineer computation spanning many resampled tokens?" The authors say they don't know yet. Can interpretability translate latent reasoning (Coconut, CoDI) back into language, and which current models are good proxies?
  - **Steering training:** can we change what is learned during fine-tuning without changing data or loss, e.g. get narrow rather than emergent misalignment? Relevant methods: CAFT, preventative steering, inoculation prompting.
  - **Model diffing:** can diffing show what a safety technique actually changed? Methods listed: activation-difference lens, logit-diff amplification, crosscoders, stage-wise diffing, activation-difference SAEs, dataset diffing.
  - **Science of misalignment:** why did a model do something concerning, and can we build a legible case for it?
  - **Monitoring:** probes as cheap first-layer monitors; hallucination probes; attention-head probes for long contexts; deception probes (Goldowsky-Dill et al.; Kretschmar et al.). Says mechanistic anomaly detection is "less explored".
  - **Proxy tasks:** build, open-source and red-team model organisms. Distill alignment faking, emergent misalignment, eval awareness and agentic blackmail into reproducible environments.
  - **Other areas:** eval-awareness suppression via steering; whether models trained with synthetic-document fine-tuning truly believe the implanted facts; introspection; CoT monitorability metrics; internals-based sandbagging detection; automated interpretability agents; data-centric interpretability (SAEs on datasets, HypotheSAEs, training-data attribution).
- **Nanda MATS stream (Summer 2026 / Winter 2027 pages)**:
  - Since mid-2024 he has been "more pessimistic about ambitious mechanistic interpretability" and does more model-biology and applied work such as monitoring.
  - Example areas: detecting deception; understanding why a model took a concerning action; stopping models from realising they are being tested.
  - Project ideas are agreed with the mentor; there is no public fixed list.
  - Sources: [MATS Nanda stream](https://www.matsprogram.org/stream/nanda); [Summer 2026 stream](https://www.matsprogram.org/stream/nanda-10)
- **Anthropic transformer-circuits.pub, 2025–2026 posts that point to open fronts:**
  - "A Toy Model of Mechanistic (Un)Faithfulness": how transcoders can go wrong (Jul 2025)
  - "A Toy Model of Interference Weights" (Jul 2025), followed by "Characterizing interference weights in a tiny language model" (Aug 2026)
  - "Sparse mixtures of linear transforms (MOLT)" (Jul 2025)
  - "Tracing Attention Computation Through Feature Interactions" (Jul 2025)
  - "Insights on Crosscoder Model Diffing" (Feb 2025)
  - "Emergent Introspective Awareness" (Lindsey, Oct 2025)
  - "Activation Oracles" cross-post (Dec 2025)
  - "Emotion Concepts…" (Apr 2026)
  - "Natural Language Autoencoders" (May 2026)
  - "Verbalizable Representations Form a Global Workspace" (Gurnee et al., Jul 2026)
  - Circuits Update Jun 2026: turn-averaged SAEs. Features found in concerning transcripts often reflect propensities such as distress or evaluation awareness.

  None of the index summaries contains an explicit "open questions" list. — [transformer-circuits.pub](https://transformer-circuits.pub/); [June 2026 update](https://transformer-circuits.pub/2026/june-update/index.html)
- **Anthropic Fellows:** fellows built and open-sourced the attribution-graph circuit-tracing method. Over 65% of fellows have published from the fellowship. — [Anthropic Fellows 2026](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)
- **Orgad, Barez, Geva et al., "Interpretability Can Be Actionable"** (arXiv 2605.11161, 11 May 2026, ICML 2026 position paper). Argues that the field's lack of impact comes from missing evaluation criteria, not missing methods. Proposes "actionability" (concreteness + validation) as the evaluation standard, across five domains. — [arXiv](https://arxiv.org/abs/2605.11161)
- **Goodfire SPD open problems:**
  - Scale to larger models; this needs better training stability.
  - SPD finds only rank-one components per layer, so a further clustering step is needed.
  - Later Goodfire work: "we lack a principled, general-purpose measure of the computational simplicity of a parameter subcomponent".

  Sources: [SPD arXiv 2506.20790](https://arxiv.org/html/2506.20790v1); [Goodfire: Interpreting LM parameters](https://www.goodfire.com/research/interpreting-lm-parameters); [goodfire-ai/spd repo](https://github.com/goodfire-ai/spd)

### Inferences
- Agenda-setters have moved toward "pragmatic", proxy-task-validated work. That favours low-compute researchers, because many proxy tasks run on open models of 8B or smaller.
- The GDM companion post is the best single source of concrete, still-open questions as of late 2025. I found no newer replacement list in 2026.

### Gaps
- I did not fetch these, so their contents are unverified here: Anthropic's "Recommended directions for AI safety research" (2025); Chris Olah's lists; Apollo, Transluce and Goodfire project-idea posts; SPAR project lists; UK AISI challenge problems.
- Sharkey et al.'s detailed subsection-level problems were not extracted. Only the abstract-level taxonomy is verified.

## Q2. Which specific open problems are tractable at small scale? (per-problem cards)

### Takeaway
At least ten concrete questions are both open in 2026 and workable with a Mac plus $500–2k of GPU. The strongest bets:
- CoT/resampling interpretability on R1-distill or Qwen3
- emergent-misalignment mechanisms, which reproduce down to 0.5B
- eval-awareness representations in open models, where the 2026 literature is fresh and contested
- model diffing with corrected crosscoders or activation-difference methods
- honest SAE-vs-baseline evaluations
- SPD scaling and clustering on toy and small models

### Cited Findings

**P1. Reverse-engineering computation spread across sampled CoT tokens in reasoning models**
- **Why open:** GDM calls reasoning-model interpretability "extremely neglected". They ask how to reverse-engineer computation spanning many resampled tokens, and say they don't know yet. Sampling is stochastic and non-differentiable, and editing CoT goes off-policy. — [GDM companion, Dec 2025](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Closest prior work:**
  - Thought Anchors (Bogdan, Macar et al.; arXiv 2506.19143). Resamples DeepSeek R1-Distill-Qwen-14B on MATH and finds that planning and uncertainty-management sentences act as anchors. The authors call sentences "a reasonable starting point", not a settled unit. — [arXiv](https://arxiv.org/html/2506.19143v4)
  - Thought Branches (ICLR 2026). In a blackmail scenario, self-preservation sentences have small causal impact. Argues for on-policy resampling over off-policy edits. — [ICLR 2026 slides](https://iclr.cc/media/iclr-2026/Slides/10008605.pdf)
- **Partial result:** a cheaper estimator of sentence importance (e.g. activation- or attention-based) that matches resampling ground truth. Or a better step-segmentation unit validated against resampling.

**P2. Steering what fine-tuning learns (narrow vs emergent misalignment) without changing data or loss**
- **Why open:** listed as an open question by GDM. Interventions are imperfect, and enough gradient pressure breaks them. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Closest prior work:** "Emergent Misalignment is Easy, Narrow Misalignment is Hard" (Soligo, Turner, Rajamanoharan, Nanda; ICLR 2026; MATS project). Findings:
  - General misalignment is linearly represented, and the representation is shared across EM fine-tunes.
  - Their model organisms are 99% coherent (vs 67% previously) and cover sizes down to 0.5B.

  Sources: [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/a79237d6c47feef24c1df723cb41f674-Paper-Conference.pdf); [MATS page](https://www.matsprogram.org/research/emergent-misalignment-is-easy-narrow-misalignment-is-hard); [Model Organisms for EM, arXiv 2506.11613](https://arxiv.org/pdf/2506.11613)
- **More 2026 work (abstracts not verified):**
  - "Characterizing the Consistency of the EM Persona" — [arXiv 2604.28082](https://arxiv.org/pdf/2604.28082)
  - "Understanding EM via Feature Superposition Geometry" — [arXiv 2605.00842](https://arxiv.org/pdf/2605.00842)
  - "Actionable Activation Directions for Detecting and Mitigating EM Across LM Families" — [arXiv 2606.20225](https://arxiv.org/pdf/2606.20225)
- **Partial result:** compare CAFT, preventative steering and inoculation prompting on the released 0.5B–8B EM organisms. Measure narrow-task retention against broad misalignment. Test robustness to more training steps.

**P3. Evaluation-awareness representations: one thing or many? Causal or correlational?**
- **Why open:** recent 2026 papers disagree.
  - Across 37 open models, probes keep AUROC ≥0.98 under rewrites that push behaviour below chance. Multi-layer steering moves downstream tasks, but single-layer injection has a "negligible" effect, including on Qwen3-8B-thinking. — [arXiv 2606.23583 "Evaluation Awareness Is Not One Capability"](https://arxiv.org/html/2606.23583v1)
  - On Qwen3-32B the link between eval awareness and refusal is "correlative", and it varies across prompts. — [arXiv 2608.27340](https://arxiv.org/html/2608.27340v1)
  - Other work: [arXiv 2608.21766 "Representation, Verbalization, and Control"](https://arxiv.org/pdf/2608.21766); [arXiv 2605.23055 "Decomposing and Measuring Evaluation Awareness"](https://arxiv.org/html/2605.23055v2); [arXiv 2608.13329 "A Probe Direction Is a Property of Its Prompt"](https://arxiv.org/pdf/2608.13329) (findings not retrieved). A probe direction from earlier work steered Qwen3-32B toward test awareness — [arXiv 2511.17085](https://arxiv.org/pdf/2511.17085).
  - GDM lists suppressing eval awareness via steering (Sonnet 4.5 system card; Hua et al.) as an area to empower. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Partial result:** a controlled test of whether eval-awareness probes track format or surface cues versus a latent belief. Or a demonstration that steering it changes misbehaviour rates on Qwen3-8B.

**P4. Model diffing: what did a fine-tune or safety technique actually change?**
- **Why open:** GDM open question (see Q1). Known artifact: L1-trained crosscoders misattribute shared concepts as unique to the fine-tuned model. Minder, Dumas, Juang, Chughtai and Nanda introduce Latent Scaling as a diagnostic and use BatchTopK as a fix. Their tests cover only Gemma 2 2B base/chat; code and models are open. — [arXiv 2504.02922](https://arxiv.org/abs/2504.02922v2); [LW write-up](https://www.greaterwrong.com/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why)
- **Related:** Anthropic "Insights on Crosscoder Model Diffing" (Feb 2025) — [transformer-circuits.pub](https://transformer-circuits.pub/). GDM reports that reasoning training mostly changed when reflexes are used, not the reflexes themselves ("base models know how to reason"). The activation-difference lens is effective on narrow fine-tunes. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Partial result:**
  - Diff a model before and after a known intervention (EM fine-tune, inoculation, refusal ablation) and check whether diffing recovers the planted change blind.
  - Extend the BatchTopK crosscoder plus Latent Scaling beyond Gemma 2 2B.

**P5. When do SAEs/transcoders beat simple baselines? ("discover unknowns" vs "act on knowns")**
- **Why open:**
  - GDM found that SAEs underperform dense linear probes on out-of-distribution harmful-intent detection. Fine-tuning SAEs on chat data closes only about half the gap. — [GDM negative results (LW)](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
  - Kantamneni et al. found no gain over linear probing across 100+ datasets. — [arXiv 2502.16681](https://arxiv.org/pdf/2502.16681)
  - A position paper argues that SAEs should be used to discover unknown concepts, not to act on known ones. — [arXiv 2506.23845](https://arxiv.org/html/2506.23845)
  - SynthSAEBench (Feb 2026) provides realistic synthetic ground truth. Matryoshka BatchTopK is the best architecture there, with probing F1 of about 0.88. — [arXiv 2602.14687](https://arxiv.org/pdf/2602.14687)
  - Feature absorption and splitting remain documented failure modes. — ["A is for Absorption", arXiv 2409.14507](https://arxiv.org/html/2409.14507v3)
  - Anthropic shows transcoders can be mechanistically unfaithful ("A Toy Model of Mechanistic (Un)Faithfulness", Jul 2025). — [transformer-circuits.pub](https://transformer-circuits.pub/)
- **Partial result:** a benchmark where the ground truth is an unknown behaviour, such as a model organism's hidden goal. Pit SAEs on Gemma Scope 2 against prompting, probing and diffing baselines.

**P6. Parameter decomposition (SPD) scaling and clustering**
- **Why open:** SPD needs better stability to scale, finds only rank-one per-layer components, and lacks a principled simplicity measure (see Q1). — [SPD paper](https://arxiv.org/html/2506.20790v1); [Goodfire](https://www.goodfire.com/research/interpreting-lm-parameters)
- **Tools:** the open repo is actively developed and accepts PRs. — [GitHub](https://github.com/goodfire-ai/spd)
- **Partial result:** a clustering method that merges rank-one components across layers on toy models. Or a stability improvement demonstrated on a tiny LM.

**P7. Probes for deception, sandbagging and hallucination: generalisation and validity**
- **Why open:** GDM lists deception monitoring as proof-of-concept stage (Goldowsky-Dill et al.; Kretschmar et al. evals), internals-based sandbagging detection as an area to develop, and attention-head probes for long contexts. It warns that naive mechanistic anomaly detection has conceptual issues. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Probe-validity worry from 2026:** "A Probe Direction Is a Property of Its Prompt". — [arXiv 2608.13329](https://arxiv.org/pdf/2608.13329)

**P8. Eliciting secret knowledge / auditing model organisms**
- **Why open:** listed as a proxy task (interpret a model organism's hidden goal). GDM also recommends building, open-sourcing and red-teaming more organisms. Cywinski et al. open-sourced secret-knowledge models. — [Pragmatic Vision](https://forum.nunosempere.com/posts/Tm2vqRZ6YMpiftCs4/a-pragmatic-vision-for-interpretability); [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Related:** Anthropic "Automated Auditing" (Jul 2025). — [transformer-circuits.pub](https://transformer-circuits.pub/)

**P9. Introspection and verbalisable representations in small open models**
- **Why open:** Anthropic's evidence comes from Claude:
  - "Emergent Introspective Awareness" (Oct 2025)
  - "Verbalizable Representations Form a Global Workspace" (Jul 2026)
  - "Activation Oracles" (Dec 2025)
  - "Natural Language Autoencoders" (May 2026)

  Sources: [transformer-circuits.pub](https://transformer-circuits.pub/). GDM lists model introspection under model psychology. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)
- **Related:** "Steering Awareness: Detecting Activation Steering from Within". — [arXiv 2511.21399](https://arxiv.org/pdf/2511.21399)
- **Partial result:** replicate the concept-injection introspection tests on Qwen3/Gemma 3 at 1–9B and map how the effect scales with size.

**P10. Do synthetic-document-fine-tuned models really "believe" implanted facts?**
- Listed as open by GDM (Slocum). — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)

**P11. Latent-reasoning models (Coconut, CoDI): translating vector CoT back to language**
- Listed as open by GDM, including which current models are good proxies. — [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)

**P12. Interference weights and attention mechanisms in tiny transformers**
- Anthropic is actively working on this (Jul 2025 toy model; Aug 2026 1-layer LM). The topic suits fully-trained tiny models. — [transformer-circuits.pub](https://transformer-circuits.pub/)

### Inferences (compute estimates are mine, not sourced)
- **P1:** resampling R1-distill-Qwen-1.5B/7B or Qwen3-8B with vLLM on one A100/H100 should cost on the order of $50–300 per study. The Mac is enough for analysis.
- **P2:** EM organisms at 0.5B–8B can be rank-1 or LoRA fine-tuned for a few dollars each, so several hundred dollars covers sweeps.
- **P3, P7:** probing and steering Qwen3-8B, Gemma-3 and Llama-3.1-8B is cheap: forward passes plus logistic regression, under $200.
- **P4:** training a crosscoder on a 2B pair is feasible within the budget. 8B pairs are borderline and need activation caching and BatchTopK.
- **P5:** reusing Gemma Scope 2 avoids SAE training cost.
- **P6, P12:** toy models train on a Mac or a single GPU.
- **P9:** introspection replication at ≤9B is cheap, but the effects may be weak or absent at small scale. A negative or scaling result is still informative.

### Gaps
- I did not verify whether P1–P12 had been fully solved by Oct 2026. Of the 2026 follow-ups in EM and eval awareness, I saw only titles and partial content.
- I did not verify specific 2026 work on the topics the brief named: developmental interpretability on Pythia/OLMo checkpoints, latent multi-hop reasoning, planning/lookahead, and steering reliability. They are not covered with sources here.
- AxBench (Wu et al. 2025, steering/concept-detection negative results) is referenced only indirectly via the position paper. Not verified.

## Q3. What open-source tooling lowers the compute bar?

### Takeaway
As of 2026, a researcher can reuse released SAEs and transcoders (Gemma Scope 2 covers all 10 Gemma 3 models), run attribution graphs locally with circuit-tracer, and browse features on Neuronpedia. Backend churn is real: the TransformerLens path versus the NNsight path for circuit-tracer.

### Cited Findings
- **Gemma Scope 2:** SAEs and transcoders for all 10 Gemma 3 models, released by GDM in Dec 2025. — [AF announcement](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2); [Neuronpedia](https://www.neuronpedia.org/gemma-scope-2)
- **Neuronpedia (Feb 2026 newsletter):** Gemma Scope 2 browsing across 10 models, circuit tracing for Gemma 3 via circuit-tracer v0.3.1, plus SAELens and NNsight updates. — [Neuronpedia blog](https://www.neuronpedia.org/blog/assistant-axis)
- **circuit-tracer:** defaults to the TransformerLens backend and has an NNsight backend. Gemma-3 transcoders from Gemma Scope 2 require NNsight. — [PyPI](https://pypi.org/project/circuit-tracer/); [GitHub](https://github.com/decoderesearch/circuit-tracer); [BlackboxNLP 2025 paper](https://aclanthology.org/2025.blackboxnlp-1.14.pdf)
- **TransformerLens 4.0:** according to Interpretune docs, it no longer supports the circuit-tracer TransformerLens path, so NNsight is the supported route there. This conflicts with the PyPI default, probably because of version differences. — [Interpretune docs](https://interpretune.org/en/latest/usage/circuit_tracer_backend_support.html)
- **Anthropic attribution-graph method:** open-sourced via the Fellows program. — [Anthropic Fellows](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)
- **HeadVis:** attention-head visualisation tool (May 2026). — [transformer-circuits.pub](https://transformer-circuits.pub/)
- **Released model organisms:** Cywinski et al. secret-knowledge models (per [GDM companion](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well)), EM organisms ([arXiv 2506.11613](https://arxiv.org/pdf/2506.11613)) and crosscoder code/models ([arXiv 2504.02922](https://arxiv.org/abs/2504.02922v2)).
- **SPD:** repo open. — [GitHub](https://github.com/goodfire-ai/spd)
- **Petri:** an automated elicitation tool from Anthropic Fellows work. — [Anthropic Fellows](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)

### Inferences
- Gemma 3 (with Gemma Scope 2) and Qwen3 / R1-distill (the reasoning-model literature) are the natural model choices in the 1–9B range.

### Gaps
- NDIF free remote access to large models: a 2025 source described it as planned. I could not confirm its current status.
- I could not confirm the exact TransformerLens version number or a v3 release.
- I did not verify open crosscoder releases beyond Minder et al.

## Q4. Recent examples of small-compute independent or MATS work with notable results

### Takeaway
MATS scholar projects with Neel Nanda have produced influential results on 0.5B–14B open models:
- emergent-misalignment organisms and the "convergent direction" (ICLR 2026)
- crosscoder artifact fixes on Gemma 2 2B
- Thought Anchors on R1-distill-Qwen-14B

This is strong evidence that the questions above are tractable at modest compute.

### Cited Findings
- **Soligo, Turner, Rajamanoharan, Nanda, "EM is Easy, Narrow Misalignment is Hard"** (ICLR 2026; MATS). Shared linear misalignment direction; organisms down to 0.5B. — [MATS](https://www.matsprogram.org/research/emergent-misalignment-is-easy-narrow-misalignment-is-hard); [paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/a79237d6c47feef24c1df723cb41f674-Paper-Conference.pdf)
- **"Model Organisms for Emergent Misalignment"** (Turner, Soligo et al.; MATS-listed). — [arXiv 2506.11613](https://arxiv.org/pdf/2506.11613); [MATS](https://www.matsprogram.org/research/model-organisms-for-emergent-misalignment)
- **Minder et al., crosscoder sparsity artifacts** (Gemma 2 2B; MATS-linked). — [arXiv 2504.02922](https://arxiv.org/abs/2504.02922v2); [MATS page](https://matsprogram.org/zh/research/overcoming-sparsity-artifacts-in-crosscoders-to-interpret-chat-tuning) (this page showed a wrong title and date, so rely on arXiv)
- **Thought Anchors** (R1-Distill-Qwen-14B, MATH). — [arXiv 2506.19143](https://arxiv.org/html/2506.19143v4)
- **Nanda's MATS 9 application** asked for about 12 hours of independent research on a problem of the applicant's choice. Listed past outcomes include EM work. — [neelnanda.io](https://www.neelnanda.io/blog/mats-apps-9)

### Inferences
- Many high-impact results came from on-the-order-of-$1k projects that built a clean model organism or proxy task, rather than from scale.

### Gaps
- I did not find concrete non-MATS independent-researcher examples (e.g. LessWrong solo posts) with verified compute costs.
- No source states the compute budgets of the projects above.
