# LLM Interpretability at Major ML/NLP Conferences and Workshops (Jul 2025 – Oct 2026)

Research date: 2026-10-15. About 20 tool calls. Coverage is uneven: the ICLR 2026 oral list and the ICML 2026 MI workshop page were read directly. NeurIPS 2025, ACL and COLM award lists could not be fully retrieved (see Gaps). Every title below was seen on an official conference page, arXiv, the ACL Anthology, or an institutional announcement, unless the entry is marked as secondary-sourced.

## Q1. Venues: awards, orals and notable interpretability papers

### Takeaway
Interpretability papers won no main-conference best-paper award at NeurIPS 2025, ICLR 2026 or ICML 2026. In all three cases the top awards went to theory, diffusion or reasoning work. Interpretability did get several ICLR 2026 orals and two EMNLP 2025 Outstanding Papers (CoT faithfulness, and causal interventions in linguistics). Most mechanistic-interpretability work appeared at dedicated workshops: MI Workshop at NeurIPS 2025 and ICML 2026, Actionable Interpretability at ICML 2025 and COLM 2026, and BlackboxNLP 2025 with its MIB shared task.

### Cited Findings

**NeurIPS 2025 (San Diego, Dec 2025)**
- NeurIPS announced its best paper awards on 2025-11-26: 4 best papers and 3 runners-up. Their topics were diffusion theory, self-supervised RL, LLM attention, LLM reasoning, online learning, scaling laws and benchmarking. None is a mechanistic interpretability paper. — [NeurIPS blog](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- **Mechanistic Interpretability Workshop @ NeurIPS 2025**
  - Held Sunday 2025-12-07 at the San Diego Convention Center, Room 30A-E. — [mechinterpworkshop.com/neurips2025](https://mechinterpworkshop.com/neurips2025/)
  - Keynotes were Chris Olah (Anthropic), Been Kim (Google DeepMind) and Sarah Schwettmann (Transluce). — [mechinterpworkshop.com/neurips2025](https://mechinterpworkshop.com/neurips2025/)
  - The CFP accepted short (4-page) and long (9-page) papers, due 2025-08-22. — [Buttondown archive](https://buttondown.com/mechinterpworkshop/archive)
  - Submissions are on OpenReview. — [OpenReview venue](https://openreview.net/submissions?venue=NeurIPS.cc%2F2025%2FWorkshop%2FMechInterp)
  - The workshop "attracted over 600 attendees", according to the ICML 2026 workshop description. — [ICML 2026 workshop page](https://icml.cc/virtual/2026/workshop/54071)
  - Background: ICML 2025 had rejected the workshop proposal, so it moved to NeurIPS. — [Neel Nanda on X](https://x.com/NeelNanda5/status/1941261748058927329)
  - The program had poster sessions and spotlight talks. — [NeurIPS schedule page](https://neurips.cc/virtual/2025/136992)
  - I found no workshop award announcement.
- Interpretability-adjacent main-track NeurIPS 2025 papers seen on neurips.cc:
  - "Theoretical Linguistics Constrains Hypothesis-Driven Causal Abstraction in Mechanistic Interpretability" — [NeurIPS](https://neurips.cc/virtual/2025/129769)
  - "MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning" — [NeurIPS](https://neurips.cc/virtual/2025/127694)
  - "Towards Multilingual Mechanistic Interpretability" — [NeurIPS](https://neurips.cc/virtual/2025/122394)
  - I could not confirm whether any of these was an oral or spotlight.

**ICLR 2026 (oral session dates ~Apr 2026)**
- Only two of the more than 5,000 accepted papers won Outstanding Paper. One is "Transformers are Inherently Succinct" (Anthony W. Lin, MPI-SWS), an expressivity/theory paper rather than interpretability. — [Saarland Informatics Campus](https://saarland-informatics-campus.de/en/piece-of-news/mpi-researcher-receives-outstanding-paper-award-at-iclr-2026/)
- Interpretability-relevant ICLR 2026 **orals**, all verified on the official oral listing ([iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)):
  - "Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability" — Bhalla, Oesterling, Verdun, Lakkaraju, Calmon (Harvard). Oral on 2026-04-23. Argues that standard SAEs find token-local features and adds temporal structure. — [ICLR oral page](https://iclr.cc/virtual/2026/oral/10008574)
  - "Verifying Chain-of-Thought Reasoning via Its Computational Graph" — Zhao, Koishekenov, Yang, Murray, Cancedda (Meta FAIR). Replaces MLPs with transcoders and builds attribution graphs over sparse features. The graph's structure predicts whether the reasoning is correct, and targeted interventions on transcoder features fix faulty reasoning. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Is it Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort" — Wang, Joshi, Plank, Angell, He He (NYU/LMU). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Interpretability" — Im, Oh, Fang, Sharon Li (UW–Madison). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology" — Fay, García-Redondo, Wang, Dubossarsky, Monod. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Temporal superposition and feature geometry of RNNs under memory demands" — Sharma, Proca, Prieto, Mediano (Imperial). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data" — Movva, Milli, Min, Pierson. Uses SAEs on preference data. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Exploratory Causal Inference in SAEnce" — Mencattini, Cadei, Locatello (ISTA). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Addressing divergent representations from causal interventions on neural networks" — Grant, Han, Tartaglini, Potts (Stanford). Takes up the interpretability-illusion and off-distribution concern about activation patching and interchange interventions. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models" — Assouel, Campbell, Bengio, Webb. A VLM mechanistic-interpretability paper. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - "Sequences of Logits Reveal the Low Rank Structure of Language Models" — Golowich, Liu, Shetty. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - Borderline orals:
    - "Hallucination Begins Where Saliency Drops"
    - "Navigating the Latent Space Dynamics of Neural Models" (Fumero, Moschella, Rodolà, Locatello)
    - "Characterizing the Discrete Geometry of ReLU Networks"
    - All three are on the [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral) listing.
- Other ICLR 2026 papers (not orals):
  - "AbsTopK: Rethinking Sparse Autoencoders…" (conference paper) — [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/527360cae239e340eb056e32b01f1caf-Paper-Conference.pdf)
  - Hanna & Ameisen, "Cross-layer transcoder feature circuits reveal latent planning in language models" — cited as ICLR 2026 in a later paper's references. Secondary only; I did not see it on an ICLR page. — [arXiv 2511.07002 references](https://arxiv.org/pdf/2511.07002)

**ICML 2026 (Seoul, COEX, Jul 6–11 2026)**
- Main awards:
  - The two Outstanding Papers were diffusion/sampling work: one on diffusion LMs and one on log-concave sampling.
  - The Outstanding Position Paper argued that alignment methods can be repurposed as a censor's toolkit.
  - The Test of Time award went to A3C (Mnih et al. 2016).
  - No interpretability paper won.
  - These details are from secondary roundups, which conflict on the announcement date (Jul 5 vs Jul 6). — [ai2.work roundup](https://ai2.work/blog/icml-seoul-roundup-diffusion-models-sweep-ai-research-awards); [besthub.dev](https://www.besthub.dev/articles/icml-2026-opens-tsinghua-wins-outstanding-paper-deepmind-earns-test-of-time-award-and-who-is-machine-learning-for-d05b9d1dd0aa)
- Interpretability position paper: "Position: Interpretability Can Be Actionable" — Orgad, Barez, Geva et al., accepted to ICML 2026. It argues interpretability should be judged by whether its insights enable concrete decisions and interventions. — [ICML poster](https://icml.cc/virtual/2026/poster/67043); [arXiv 2605.11161](https://arxiv.org/abs/2605.11161)
- Critique paper at ICML 2026: "Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation" — [ICML 2026](https://icml.cc/virtual/2026/69389)
- **Mechanistic Interpretability Workshop @ ICML 2026** (third edition)
  - Held Friday 2026-07-10, Hall C, COEX Seoul. — [ICML workshop page](https://icml.cc/virtual/2026/workshop/54071)
  - Organizers: Neel Nanda (GDM), Andrew Lee (Harvard), Andy Arditi (Northeastern), Stefan Heimersheim, Anna Soligo (Imperial), J Rosser (Oxford), Iván Arcuschin (Poseidon Research), Taero Kim and Sungjun Lim (Yonsei). Sponsored by Poseidon Research. — [mechinterpworkshop.com](https://mechinterpworkshop.com/)
  - About 350 accepted papers are listed, including 23 spotlights. — [ICML workshop page](https://icml.cc/virtual/2026/workshop/54071); [mechinterpworkshop.com](https://mechinterpworkshop.com/)
  - The stated aim is to encourage debate between "pragmatic and ambitious" approaches. — [ICML workshop page](https://icml.cc/virtual/2026/workshop/54071)
  - Invited lightning talks (from the [ICML workshop page](https://icml.cc/virtual/2026/workshop/54071)):
    - "Dissociating the Internal Representations of Sycophancy in LLMs" (Baez et al.)
    - "Learn and Steer: Mechanistically Informed Interventions for Discrete Diffusion Language Models" (Zhou et al.)
    - "Towards Measuring and Detecting Unverbalized Evaluation Awareness" (Montoya et al.)
    - "Critical Percolation as a Synthetic Data Model for Interpretability" (Brill, Carlson)
    - "Before the Last Token: Diagnosing Final-Token Safety Probe Failures" (Doda)
    - "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads" (Gema et al.)
    - "Tracking Training Phases in Compositional Learning with Task-Agnostic Measures" (Dern et al.)
    - "The Connectome of a Large Language Model" (Deng, Jin)
    - "What is Missing? Explaining Neurons Activated by Absent Concepts" (Hesse et al.)
    - "Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism" (Orgad et al.)
- 2nd Workshop on Compositional Learning: Safety, Interpretability, and Agents (ICML 2026, Jul 10) included a Best Paper Awards session. I could not retrieve the winners. — [ICML](https://icml.cc/virtual/2026/80468)

**ACL 2025 (Vienna, Jul–Aug 2025)**
- I found no ACL 2025 best paper on interpretability.
- ACL 2025 Outstanding Paper: "Mapping 1,000+ Language Models via the Log-Likelihood Vector" (Oyama, Yamagiwa, Takase, Shimodaira; Kyoto/RIKEN). This is model characterization, not internals. — [Kyoto Univ.](https://stat.sys.i.kyoto-u.ac.jp/?p=1961); [RIKEN AIP](https://aip.riken.jp/?p=20603)
- Related, but a different venue: NAACL 2025 gave its SAC Award for Interpretability to "On Behalf of the Stakeholders: Trends in NLP Model Interpretability in the Era of LLMs" (Calderon & Reichart, Technion). — [NAACL 2025 best papers](https://2025.naacl.org/blog/best-papers/)

**EMNLP 2025 (Suzhou, Nov 2025)**
- Outstanding Paper: "Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps" — Tutek, Belinkov et al. (Technion). — [Technion CS news](https://www.cs.technion.ac.il/news/view-new.php?nwid=1581)
- Outstanding Paper: "Causal interventions reveal shared structure across English filler-gap constructions" — Boguraev, Potts, Mahowald (UT Austin/Stanford). — [Potts CV](https://stanford.edu/~cgpotts/cv)
- MRL 2025 workshop Best Paper: "TenseLoC: Tense Localization and Control in a Multilingual LLM" (Saarland). — [Saarland news](https://www.uni-saarland.de/en/department/lst/news/best-paper-award-mrl-2025-ws-at-emnlp-40784.html)

**BlackboxNLP 2025 (8th edition, co-located with EMNLP 2025, Suzhou, Nov 2025)**
- The shared task extended **MIB (Mechanistic Interpretability Benchmark)** with two tracks: circuit localization and causal-variable localization.
  - On circuit localization, 3 teams with 8 methods made notable gains using ensembling and regularization.
  - On causal-variable localization, 1 team made gains using low-dimensional and non-linear projections.
  - The leaderboard remains open.
  - Proceedings pp. 543–552. — [arXiv 2511.18409](https://arxiv.org/abs/2511.18409v1); [ACL Anthology](https://preview.aclanthology.org/setup/2025.blackboxnlp-1.32)
- Participant paper: "BlackboxNLP-2025 MIB Shared Task: Improving Circuit Faithfulness via Better Edge Selection" (Nikankin et al.; Belinkov group). Uses bootstrapping to find edges with consistent attribution scores. — [arXiv 2510.25786](https://arxiv.org/abs/2510.25786v1)
- The MIB paper itself is arXiv 2504.13151. It was presented at ICML 2025, which falls just before this window. — [arXiv](https://arxiv.org/pdf/2504.13151)

**COLM 2025 (Montreal, Oct 7–10 2025) and COLM 2026**
- An interpretability paper accepted at COLM 2025: "Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B" (Bakalova, Veitsman, Huang, Hahn; Saarland). — [Saarland news](https://www.uni-saarland.de/fachrichtung/lst/news/wissenschaftliche-beitraege-beim-colm-2025-angenommen-38726.html)
- **Actionable Interpretability Workshop @ COLM 2026**
  - Held 2026-10-09 in San Francisco, the second edition after ICML 2025.
  - Organizers: Haklay (Goodfire), Orgad (Kempner/Harvard), Reusch (Technion), Mosbach (McGill/Mila), Wiegreffe (UMD), Tenney (GDM), Geva (TAU), Avrahamy (Meta FAIR/TAU).
  - Keynotes:
    - Tom McGrath (Goodfire), "Ambitious Actionable Interpretability"
    - Dhanya Sridhar (Mila)
    - Yonatan Belinkov (Technion)
    - Christopher Potts (Stanford), "Post Hoc, Ergo Propter Data"
  - Contributed talks:
    - "Shifting Mechanisms: How Positional Encoding Choice Shapes Long-Context Retrieval"
    - "Self-CTRL"
    - "How Faithful Is Trajectory-Based Data Attribution?"
    - "Verbalizing LLMs' assumptions to explain and control sycophancy…"
  - Source: [actionable-interpretability.github.io](https://actionable-interpretability.github.io/)
  - An aggregator reports 71 accepted papers (unverified). — [digg](https://digg.com/ai/z6mi5k2a)
- ICML 2025 Actionable Interpretability workshop (first edition) included a talk on NDIF/NNsight. — [ICML 2025](https://icml.cc/virtual/2025/51374)

### Inferences
- Mechanistic interpretability is under-represented among main-track top awards. The awards that went to interpretability-adjacent work (EMNLP CoT faithfulness, filler-gap causal interventions) came from NLP venues and favor causal-intervention or faithfulness framings.
- The MI workshop grew from 140+ papers at ICML 2024 to ~350 accepted at ICML 2026, with 600+ attendees at NeurIPS 2025. That growth points to a fast-expanding community whose output mostly lands at workshops.

### Gaps
- I could not find a full list of NeurIPS 2025 interpretability orals or spotlights, or any MI Workshop award winners. No award announcement was found, and the main site lists none.
- I could not get the full ACL 2025, EMNLP 2025, COLM 2025 or BlackboxNLP 2025 best-paper lists. ACL 2026 and COLM 2026 main-track awards were not retrieved.
- ICML 2026 MI workshop spotlight titles (23) and any prize winners were not itemized on the pages I fetched.
- The ICLR 2026 oral list was read in four chunks by a summarizer and could still miss a title.

## Q2. Dominant themes

### Takeaway
The dominant themes in this period were:
- refinements and critiques of SAEs (temporal SAEs, AbsTopK, SAE vs probe comparisons)
- transcoder and attribution-graph circuit tracing applied to reasoning verification
- CoT faithfulness and monitorability
- rigorous causal-intervention evaluation (MIB, divergent-representation critiques)
- representation and feature geometry
- a growing "actionable/pragmatic" turn: steering, sycophancy, evaluation awareness and safety probes

### Cited Findings
- **SAE variants**: Temporal SAEs (ICLR 2026 oral) — [ICLR](https://iclr.cc/virtual/2026/oral/10008574); AbsTopK (ICLR 2026) — [proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/527360cae239e340eb056e32b01f1caf-Paper-Conference.pdf)
- **SAEs used as analysis tools** on preference data and for causal inference (ICLR 2026 orals). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
- **Transcoders and attribution graphs**:
  - CRV, "Verifying CoT Reasoning via Its Computational Graph" (ICLR 2026 oral). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - The method builds on Anthropic's cross-layer-transcoder circuit tracing. — [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
  - Follow-on arXiv work: ADAG, which automatically describes attribution graphs — [arXiv 2604.07615](https://arxiv.org/pdf/2604.07615); CLT-Forge library — [arXiv 2603.21014](https://arxiv.org/html/2603.21014v1). Venue not verified for either.
- **CoT faithfulness and monitorability**:
  - Faithfulness via unlearning reasoning steps (EMNLP 2025 Outstanding). — [Technion](https://www.cs.technion.ac.il/news/view-new.php?nwid=1581)
  - Detecting implicit reward hacking through reasoning effort (ICLR 2026 oral). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - Unverbalized evaluation awareness (ICML 2026 MI workshop talk). — [ICML](https://icml.cc/virtual/2026/workshop/54071)
- **Causal-abstraction evaluation and benchmarks**:
  - MIB shared task at BlackboxNLP 2025. — [arXiv 2511.18409](https://arxiv.org/abs/2511.18409v1)
  - "Addressing divergent representations from causal interventions" (ICLR 2026 oral). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - Theoretical-linguistics constraints on causal abstraction (NeurIPS 2025). — [NeurIPS](https://neurips.cc/virtual/2025/129769)
- **Feature and representation geometry**:
  - Temporal superposition in RNNs, persistent homology of latent spaces, and the low-rank logit structure (all ICLR 2026 orals). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
- **Multimodal/VLM interpretability**: "Visual symbolic mechanisms" (ICLR 2026 oral). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
- **Safety-relevant mechanisms and probes** (all ICML 2026 MI workshop talks):
  - a distinct, unified mechanism for harmful content
  - final-token safety probe failures
  - sycophancy representations
  - Source: [ICML](https://icml.cc/virtual/2026/workshop/54071)
- **Interpretability of non-autoregressive models**: steering discrete diffusion LMs (ICML 2026 MI workshop talk). — [ICML](https://icml.cc/virtual/2026/workshop/54071)
- **The "actionable" framing**:
  - ICML 2026 position paper. — [arXiv 2605.11161](https://arxiv.org/abs/2605.11161)
  - COLM 2026 workshop. — [site](https://actionable-interpretability.github.io/)
  - The ICML 2026 MI workshop explicitly frames a "pragmatic vs ambitious" debate. — [ICML](https://icml.cc/virtual/2026/workshop/54071)
- **Steering**: interpretability scores are a weak proxy for steering utility. — [arXiv 2510.03659](https://arxiv.org/html/2510.03659v1) (venue not verified)

### Inferences
- The field has moved from "find features" toward "use internals to verify, monitor or intervene", especially on reasoning traces. CRV, the reward-hacking detector and CoT faithfulness work all point that way.

### Gaps
- I found no venue-verified conference papers on crosscoders/model diffing or on parameter-space decomposition (APD/SPD) in this window. Those may exist at workshops but were not confirmed.
- I found no verified conference-venue paper on steering-vs-prompting comparisons or deception probes.

## Q3. Prominent academic groups

### Takeaway
Israeli NLP-interpretability groups are prominent, with Technion (Belinkov) and Tel Aviv (Geva) co-running MIB, the BlackboxNLP shared task and Actionable Interpretability. Stanford (Potts) leads the causal-intervention/abstraction line. Harvard (Kempner; Lakkaraju/Calmon) works on SAE methods and actionable interpretability. Northeastern (Bau lab, through Arditi and NDIF) and Imperial/Oxford people organize the MI workshop together with GDM's Nanda.

### Cited Findings
- **Technion, Belinkov**:
  - EMNLP 2025 Outstanding Paper on CoT faithfulness. — [Technion](https://www.cs.technion.ac.il/news/view-new.php?nwid=1581)
  - MIB shared-task entry (Nikankin et al.). — [arXiv 2510.25786](https://arxiv.org/abs/2510.25786v1)
  - COLM 2026 keynote. — [site](https://actionable-interpretability.github.io/)
  - Technion's Calderon & Reichart won the NAACL 2025 interpretability SAC award. — [NAACL](https://2025.naacl.org/blog/best-papers/)
- **Tel Aviv, Geva; Harvard Kempner, Orgad**:
  - "Interpretability Can Be Actionable" (ICML 2026). — [ICML](https://icml.cc/virtual/2026/poster/67043)
  - Organizers of the Actionable Interpretability workshop. — [site](https://actionable-interpretability.github.io/)
  - Orgad et al.'s harmful-content mechanism talk at the ICML 2026 MI workshop. — [ICML](https://icml.cc/virtual/2026/workshop/54071)
- **Stanford, Potts**:
  - ICLR 2026 oral on divergent representations from causal interventions. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - EMNLP 2025 Outstanding Paper on filler-gap causal interventions. — [Potts CV](https://stanford.edu/~cgpotts/cv)
- **Harvard, Lakkaraju and Calmon**: Temporal SAEs, ICLR 2026 oral. — [ICLR](https://iclr.cc/virtual/2026/oral/10008574)
- **Northeastern / NDIF (Bau lab)**:
  - Andy Arditi (Northeastern) co-organizes the MI workshop. — [mechinterpworkshop.com](https://mechinterpworkshop.com/)
  - NDIF/NNsight talk at the ICML 2025 Actionable Interpretability workshop. — [ICML 2025](https://icml.cc/virtual/2025/51374)
- **Imperial**:
  - Mediano group, ICLR 2026 oral on temporal superposition. — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - Soligo co-organizes the MI workshop. — [mechinterpworkshop.com](https://mechinterpworkshop.com/)
- **Saarland, Hahn**: COLM 2025 ICL circuits paper; MRL 2025 best paper. — [Saarland](https://www.uni-saarland.de/fachrichtung/lst/news/wissenschaftliche-beitraege-beim-colm-2025-angenommen-38726.html); [Saarland](https://www.uni-saarland.de/en/department/lst/news/best-paper-award-mrl-2025-ws-at-emnlp-40784.html)
- **Industry at academic venues**:
  - Meta FAIR's CRV (ICLR 2026 oral). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
  - Goodfire co-organizes and keynotes Actionable Interpretability. — [site](https://actionable-interpretability.github.io/)
  - Anthropic, GDM and Transluce gave the NeurIPS 2025 MI workshop keynotes. — [site](https://mechinterpworkshop.com/neurips2025/)

### Inferences
- Organizers and authors increasingly straddle academia and interpretability startups such as Goodfire and Transluce. That suggests a joint academic–industry pipeline rather than separate communities.

### Gaps
- I gathered no venue-verified 2025–26 papers specifically from MIT/Tegmark, Oxford or the Geiger/Pr(Ai)²R group in this pass. They are probably present, but I did not confirm any.

## Q4. Critiques and negative results that gained traction

### Takeaway
The most influential negative result is that SAE-based probes fail to beat simple linear probes on downstream tasks, especially out of distribution. GDM's interpretability team deprioritized fundamental SAE research in 2025 for this reason. 2026 work extends the critique to compositional generalization and argues the failures stem from amortized encoders rather than from superposition itself. Separately, causal-intervention "illusions" (off-distribution representations) were addressed at the ICLR 2026 oral level.

### Cited Findings
- Kantamneni et al., "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (with Nanda and Rajamanoharan). SAE probes underperform logistic regression on average across regimes. — [arXiv 2502.16681](https://arxiv.org/html/2502.16681v1)
- GDM's MI team reported that SAEs underperformed linear probes on OOD harmful-intent detection and deprioritized fundamental SAE research (March 2025). This is from a secondary summary. — [LongtermWiki](https://www.longtermwiki.com/resources/244c1b93ef0a083c)
- "Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation" (ICML 2026). — [ICML](https://icml.cc/virtual/2026/69389)
- A Mila 2026 counterpoint argues the OOD failures come from a "brittle, amortised encoder", not from the superposition hypothesis. — [arXiv 2603.28744](https://arxiv.org/html/2603.28744v1)
- "Do Sparse Autoencoders Generalize? A Case Study of Answerability". — [arXiv 2502.19964](https://arxiv.org/pdf/2502.19964)
- "Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders" (arXiv 2606.12138). Venue not verified. — [arXiv](https://arxiv.org/pdf/2606.12138)
- Interventions can produce divergent, off-distribution representations (ICLR 2026 oral, Potts group). — [iclr.cc orals](https://iclr.cc/virtual/2026/events/oral)
- Safety probes that read only the final token fail ("Before the Last Token", ICML 2026 MI workshop). — [ICML](https://icml.cc/virtual/2026/workshop/54071)
- Interpretability scores weakly predict steering utility. — [arXiv 2510.03659](https://arxiv.org/html/2510.03659v1)

### Inferences
- The SAE critique has driven two reactions: architectural fixes (temporal SAEs, AbsTopK, transcoders/CLTs) and a pivot toward "actionable" or pragmatic evaluation, where methods are judged by downstream task performance against simple baselines.

### Gaps
- I could not access the primary GDM Alignment Forum post. The specifics rest on a secondary summary.
- I did not verify whether Kantamneni et al. appeared at ICML 2025, since that predates this window.
