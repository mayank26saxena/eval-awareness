# Is this a good project? Honest review (2026-10-15)

## Verdict
**Yes, but with a revised framing.** The topic is important, and every major lab names it as an unsolved problem. Some parts of the original plan **have already been done**, though. The 2×2 format-vs-context dataset is not new anymore. The new parts are:
1. checking whether **activation readers (NLAs, Activation Oracles) actually read the model or just read the prompt**
2. a **causal test of the format-removed direction**

This area moves fast: at least 6 papers appeared in 2026. Move quickly and post early.

---

## 1. Novelty check: what's already been done

| Piece of our plan | Status | Who did it |
|---|---|---|
| Probes separate eval from deploy | ✅ done | Nguyen et al. 2025 (2507.01786) |
| **2×2 format × context dataset** | ⚠️ **done** | Devbunova, Mar 2026, ICLR workshop (2603.19426): probes mostly track benchmark format. "Not One Capability", Jun 2026 (2606.23583): 2×2 design, yet probes survive rewrites (AUROC ≥0.98) |
| Realistic rewrites change behavior | ✅ done | Probe-Rewrite-Evaluate (2509.00591): deception −25%. Anthropic NLA paper. GDM realistic honeypots |
| Steering suppresses eval awareness | ✅ done | Hua et al. 2025; Sonnet 4.5 system card; "Not One Capability" (3-layer steering works); 2609.22119 (counter-steering flips 71%) |
| NLA detects eval awareness | ✅ done **on Claude only** | Anthropic: 26% of SWE-bench Verified vs <1% of real traffic. **No ground truth**, as the authors admit |
| **NLAs/AOs on open models for eval awareness** | 🟢 **not found** | — |
| **Text-only vs activation-reader detectors on format-controlled data** | 🟢 **not found** | — |
| **Steering with a direction that has format removed (LEACE)** | 🟡 partly | Others steer a mixed eval/deploy direction. None removes format first |
| PII as the behavior | 🟡 minor novelty | Others use refusal, deception, sandbagging |

**What this means:** the field has **contradictory results** (Devbunova: "it's just format" vs "Not One Capability": "the signal survives rewrites"). Anthropic's NLA eval-awareness claim **lacks ground truth**. Your project can address both.

## 2. The revised research question
> **"When an activation reader says 'the model thinks it's being tested', is it reading the model's mind or just the prompt's format? And does the format-free part of that signal causally drive behavior?"**

Contributions, in priority order:
1. **First ground-truth test of NLAs and Activation Oracles for eval awareness.** Use the controlled 2×2 to see whether they follow format or the actual context. This directly checks an Anthropic claim using their released tools.
2. **Settle the Devbunova vs "Not One Capability" disagreement**: remove format with LEACE, then steer, and see whether anything causal remains.
3. A text-only vs activation-reader detector comparison (Jev, LLM judge, probe, AO, NLA).

## 3. Weak spots in the original idea (and fixes)

| Problem | Why it matters | Fix |
|---|---|---|
| **"Belief" isn't directly observable.** A casual-looking eval still contains *some* cues (fake company names, harness hints) | "Format vs belief" is really "surface format cues vs contextual cues". Overclaiming "belief" will get criticized | Frame it as **format cues vs contextual cues**. Systematically vary which contextual cues are present |
| **12B may be too small.** "Not One Capability" finds ≤2B models near chance and the best AUROC only 0.71 at 14B. 2609.22119 says small models rely on format | At 12B you might just confirm "it's format", a weaker result | **Develop on Gemma-3-12B, headline on Gemma-3-27B.** It has an NLA (layer 41) and Gemma Scope 2. Llama-3.3-70B (also has an NLA) is the stretch goal |
| **PII leaks may be rare** in a toy sandbox | A small behavior gap means low statistical power | Main behavior: **honesty/deception** (PRE shows a 25% swing) or sandbagging. Keep PII as the Amazon-relevant secondary behavior |
| **Scoop risk** | 6+ papers in 2026 | Cut to about 6 weeks. Post an early Alignment Forum note after the NLA/AO ground-truth result |
| NLA / AO "text inversion" | They may infer from context | This is what you measure, so it's a feature, not a bug |

## 4. Impact assessment

| Dimension | Rating | Why |
|---|---|---|
| Importance | ⭐⭐⭐⭐⭐ | Anthropic, OpenAI/Apollo, GDM, UK AISI and METR all say eval awareness undermines safety evals |
| Novelty (revised) | ⭐⭐⭐½ | The NLA/AO ground-truth test and the format-removed causal test are new. The 2×2 dataset alone is not |
| Feasibility | ⭐⭐⭐⭐ | Open checkpoints exist. Budget is about $600–1,000. Main risk is behavior gaps that are too small |
| Realistic outcome | Workshop paper / AF post | Credible portfolio work for MATS / Anthropic Fellows / an internal Amazon pitch |
| Who uses it | Eval designers, interpretability tool builders, Nova safety team | — |

## 5. Practical takeaways (whatever the result)
1. **For eval designers:** if behavior follows format, *rewrite evals to look realistic* (cheap). If it follows context, rewriting won't work, so use *production-derived honeypots* and internal monitoring.
2. **For interpretability tool users:** a probe or NLA reading of "eval-aware" may just be a prompt-format detector. **Always validate on format-controlled data before trusting it.**
3. **For steering-based mitigations** (Anthropic, Hua et al.): removing format first shows whether steering removes awareness or just shifts format.
4. **For Amazon / Nova:** a cheap audit is to run the 2×2 behavior test on Nova. A large gap means safety-eval numbers overstate deployment safety, and that needs fixing before trusting the evals.
5. **For you:** the dataset, the detector-comparison harness and the LEACE+steering pipeline are reusable for other "does the model know X?" questions (sycophancy, sandbagging).

## 6. Go / no-go
**Go**, with the changes above. If you only have time for one result, do **contribution #1** (the NLA/AO ground-truth test on Gemma-3-12B/27B). It's the most novel, the cheapest, and has a clear story whichever way it comes out.

## Sources
- [Devbunova, "Is Evaluation Awareness Just Format Sensitivity?" (2603.19426)](https://arxiv.org/abs/2603.19426)
- [Evaluation Awareness Is Not One Capability (2606.23583)](https://arxiv.org/html/2606.23583v1)
- [Probe-Rewrite-Evaluate (2509.00591)](https://arxiv.org/html/2509.00591v6)
- [Format→Context with scale (2609.22119)](https://arxiv.org/abs/2609.22119)
- [Representation, Verbalization, and Control (2608.21766)](https://arxiv.org/abs/2608.21766)
- [Decomposing and Measuring Evaluation Awareness (2605.23055)](https://arxiv.org/html/2605.23055v2)
- [Anthropic NLA paper](https://transformer-circuits.pub/2026/nla/) · [Anthropic NLA post](https://www.anthropic.com/research/natural-language-autoencoders)
- [Global Workspace (2607.15495)](https://arxiv.org/pdf/2607.15495)
- [When Activation Oracles Learn Not to Read (2607.23379)](https://arxiv.org/pdf/2607.23379)
- Industry sources: see `references/landscape.md`

## Update 2026-10-25
EvalAwareBench (aisa-group, arXiv 2605.23055) already varies 8 eval-cue factors independently, but measures only black-box CoT verbalization. Our dataset borrows its taxonomy. The novel parts are unchanged: activation-based detectors checked against controlled factors, the NLA grounding test, and causal steering.
