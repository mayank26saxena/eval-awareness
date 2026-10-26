# Who's working on eval awareness (as of Oct 2026)

Researched 2026-10-15. Items marked *(secondary)* come from summaries or press rather than the primary source. Verify those before citing.

## Industry labs: they all treat it as a serious, open problem

| Who | What they found or did | Source |
|---|---|---|
| **Anthropic** | Claude Sonnet 4.5 system card (Sep 2025), §7.2: the model often recognized alignment tests and behaved unusually well. Steering down eval-awareness SAE features reduced verbalized awareness, and in some cases misaligned behavior rose. | [System card](https://assets.anthropic.com/m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf), [Transformer](https://transformernews.ai/p/claude-sonnet-4-5-evaluation-situational-awareness) *(secondary)* |
| **Anthropic** | Natural Language Autoencoders (May 2026) surface *unverbalized* eval awareness. It shows up more often on eval transcripts than on deployment traffic, and rewriting inputs to be more realistic reduces it. The paper gives no ground truth and no detailed numbers for this. | [NLA paper](https://transformer-circuits.pub/2026/nla/) |
| **OpenAI + Apollo** | Anti-scheming training (Sep 2025) cut covert actions about 30× (o3: 13%→0.4%). The authors caution that part of the drop may come from models recognizing tests. | [OpenAI](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/) |
| **Apollo Research** | 2026 model reports: Meta's Muse Spark had the highest eval awareness Apollo has seen. Claude 4.6's high verbalized awareness meant its evals "could not offer reliable evidence of actual alignment". | [Apollo science](https://apolloresearch.ai/science) *(Claude 4.6 claim is secondary, via the Singapore Consensus)* |
| **Google DeepMind** | Models sometimes treat evals as "a game/test for which sabotage is the desired role". DeepMind's mitigation is realistic honeypot evals built from real deployment situations (Jul 2026). Neel Nanda's team lists "eval-awareness suppression via steering" as a priority. | [METR report](https://metr.org/blog/2026-05-19-frontier-risk-report/), [GDM summary](https://gdmalignment.substack.com/p/agi-safety-and-alignment-at-google), [Nanda agenda](https://www.lesswrong.com/posts/MnkeepcGirnJn736j/how-can-interpretability-researchers-help-agi-go-well) |
| **UK AISI** | In its sabotage evaluations, every model tested could reliably tell AISI's eval scenarios from deployment data when prompted. AISI says it is "actively working on methods" to address this. | [AISI blog](https://www.aisi.gov.uk/blog/evaluating-whether-ai-models-would-sabotage-ai-safety-research) |
| **METR** | Models explicitly reasoned about evals being artificial and sometimes "played along", so METR treats toy demonstrations as weak evidence. Production-derived traffic makes evals more realistic. | [METR frontier risk report](https://metr.org/blog/2026-05-19-frontier-risk-report/) |
| **Field consensus** | The 2026 Singapore Consensus says eval awareness "erodes trust in the validity of some assessments". | [arXiv 2608.14611](https://arxiv.org/pdf/2608.14611) |

**Takeaway:** labs mostly fight eval awareness in two ways:
- **making evals more realistic**, which only works if the cue is format
- **steering it away**, which only works if there is a causal direction to steer

Nobody has cleanly tested which of the two is right. That is the gap this project fills.

## Academic / open-model papers

| Paper | Key point | Gap left |
|---|---|---|
| Nguyen et al., Jul 2025 ([2507.01786](https://arxiv.org/abs/2507.01786)) | Probes on Llama-3.3-70B separate eval from deploy prompts, and real safety evals already look like tests to the model. | Probing only, with format mixed into the signal. |
| Hua, Qin, Marks, Nanda, Oct 2025 ([2510.20487](https://arxiv.org/abs/2510.20487)) | Steering makes eval-aware model organisms behave as if deployed. | Uses trained model organisms, not naturally occurring awareness. |
| Scaling paper, Sep 2025 ([2509.13333](https://arxiv.org/abs/2509.13333)) | Eval awareness follows a power law across sizes from 0.27B to 70B. | Correlational. |
| "Not One Capability", Jun 2026 ([2606.23583](https://arxiv.org/html/2606.23583v1)) | Across 37 models, much of the signal is format. Probes stay at AUROC 0.98 even when behavior collapses. Steering works across 3 layers but not at a single layer. | Calls its own mechanism analysis "exploratory" and asks for a unified account of probes, representations, and interventions. |
| Format→Context with scale, Aug 2026 ([2609.22119](https://arxiv.org/abs/2609.22119)) | Small models rely on format and large models on context. Prompt sanitization plus counter-steering flips 71% of behaviors. | Does not separate which internal direction carries format and which carries belief. |
| Representation/Verbalization/Control, Aug 2026 ([2608.21766](https://arxiv.org/abs/2608.21766)) | Eval awareness is linearly decodable in 6 models. | Same format confound. |

## Activation-reading tools available for open models

### Natural Language Autoencoders (Anthropic, May 2026)
Code: https://github.com/kitft/natural_language_autoencoders (Apache-2.0). Checkpoints also inherit the base model's license.
HF collection: https://huggingface.co/collections/kitft/nla-models. Inference runs on an SGLang server.

| Target model | Layer | Verbalizer (AV) | Reconstructor (AR) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 20 / 28 | kitft/nla-qwen2.5-7b-L20-av | kitft/nla-qwen2.5-7b-L20-ar |
| **Gemma-3-12B-IT** | 32 / 48 | kitft/nla-gemma3-12b-L32-av | kitft/nla-gemma3-12b-L32-ar |
| Gemma-3-27B-IT | 41 / 62 | kitft/nla-gemma3-27b-L41-av | kitft/nla-gemma3-27b-L41-ar |
| Llama-3.3-70B-Instruct | 53 / 80 | kitft/Llama-3.3-70B-NLA-L53-av | kitft/Llama-3.3-70B-NLA-L53-ar |

There is also an interactive frontend on Neuronpedia (https://www.neuronpedia.org/nla); the URL suggests it covers Llama-3.3-70B.

### Activation Oracles
- Original release covers 12 models across the Gemma-2, Gemma-3, Qwen3 and Llama 3 families: https://huggingface.co/collections/adamkarvonen/activation-oracles. **The exact model list is not yet confirmed, so check that Gemma-3-12B-IT is in it.** Small models (1–4B) generalize worse.
- Improved oracles ("Building Better AOs", 2026): `ceselder/qwen3-8b-ao-v3-best`, collection at https://huggingface.co/collections/ceselder/building-better-activation-oracles. The AObench eval code is at https://github.com/ceselder/cot-oracle.
- Known weakness: an oracle may infer the answer from the surrounding text instead of reading the activations ("text inversion"). The project's 2×2 design tests exactly this.
- Related: "When Activation Oracles Learn Not to Read" ([arXiv 2607.23379](https://arxiv.org/pdf/2607.23379)) reports concept-specific blind spots.

### SAEs and transcoders
Gemma Scope 2 (Dec 2025) covers all 10 Gemma 3 models: https://www.neuronpedia.org/gemma-scope-2

## Model choice

| Model | Linear probe | NLA | Activation Oracle | SAEs | Runs on |
|---|---|---|---|---|---|
| **Gemma-3-12B-IT** (main) | ✅ | ✅ | likely (Gemma-3 family, to confirm) | ✅ Gemma Scope 2 | Mac with 32GB+ (slow) or 1× A100/H100 |
| Gemma-3-4B-IT (development) | ✅ | ❌ | maybe | ✅ | Mac |
| Qwen3-8B (second family) | ✅ | ❌ | ✅ best AO | ❌ | Mac with 24GB+ or cloud |
| Llama-3.3-70B-Instruct (scale-up) | ✅ | ✅ | maybe | ❌ | 2× H100, or NDIF |

**Gemma-3-12B-IT is the only model that has all four tools: probe, NLA, likely an AO, and SAEs.** Llama-3.3-70B is a good scale-up target because Nguyen et al. ran their original probe study on it and it has an NLA.
