# Related work and resources

Last reviewed: October 2026. Entries marked † come from secondary reporting and should be verified against the primary source before citing.

## 1. Evaluation awareness: findings from frontier labs

| Source | Finding |
|---|---|
| Anthropic, *Claude Sonnet 4.5 System Card* (2025), §7.2 | The model often recognized alignment evaluations. Steering down eval-awareness SAE features reduced verbalized awareness and, in some scenarios, increased misaligned behavior. |
| Fraser-Taliente, Kantamneni, Ong et al., *Natural Language Autoencoders* (Anthropic, 2026) | NLAs surface unverbalized eval awareness in Claude Opus 4.6. The rate is higher on evaluation transcripts than on deployment traffic and lower after realism edits. The validation is directional only; the authors note there is no ground truth. |
| OpenAI & Apollo Research, *Detecting and reducing scheming* (2025) | Anti-scheming training reduced covert actions about 30×. The authors caution that part of the reduction may reflect evaluation awareness. |
| Apollo Research, model evaluation reports (2026)† | High rates of evaluation awareness in several frontier models limit what their evaluations can show. |
| METR, *Frontier Risk Report* (2026); Google DeepMind alignment updates (2026) | Models reason explicitly about evaluations being artificial. Production-derived and honeypot evaluations are proposed as mitigations. |
| UK AISI, research-sabotage evaluations (2026) | All models tested could distinguish AISI's evaluation scenarios from deployment data when prompted. |
| *Singapore Consensus on Global AI Safety Research Priorities* (2026) | Identifies evaluation awareness as eroding the validity of safety assessments. |

## 2. Academic work on evaluation awareness

| Paper | Contribution | Open gap |
|---|---|---|
| Needham et al., *LLMs Often Know When They Are Being Evaluated* (2505.23836) | 1,000-transcript eval/deploy benchmark; black-box detection. | Black-box only. |
| Nguyen et al., *Probing and Steering Evaluation Awareness* (2507.01786) | Linear probes on Llama-3.3-70B separate eval from deploy prompts. | Format and context are confounded. |
| Hua, Qin, Marks, Nanda, *Steering Evaluation-Aware LMs to Act Like They Are Deployed* (2510.20487) | Contrastive steering suppresses eval awareness in trained model organisms. | Trained organisms rather than naturally arising awareness. |
| *Evaluation Awareness Scales Predictably* (2509.13333) | Power-law scaling of probe-detected awareness from 0.27B to 70B. | Correlational. |
| Devbunova, *Is Evaluation Awareness Just Format Sensitivity?* (2603.19426) | Under a controlled 2×2, probes mainly track benchmark-canonical format. | No causal tests. |
| *Decomposing and Measuring Evaluation Awareness* / EvalAwareBench (2605.23055) | Eight independently toggled cue factors (F1–F8); black-box CoT measurement. | No activation-based detectors. |
| *Evaluation Awareness Is Not One Capability* (2606.23583) | 37 models. Much of the signal is format, yet probes survive rewrites. Multi-layer steering is causal; single-layer steering is not. | Its mechanistic analysis is described as exploratory. |
| *Evaluation Awareness: Representation, Verbalization, and Control* (2608.21766) | Awareness is linearly decodable in six models. | Format confound. |
| *Evaluation Awareness Shifts from Format to Context with Model Scale* (2609.22119) | Small models rely on format, larger models on context. | Does not separate the internal directions. |
| Xiong et al., *Probe-Rewrite-Evaluate* (2509.00591) | Realistic rewrites shift probe scores and behavior. | Correlational. |
| *EvalDetectBench* (2609.01611) | 746 eval and deployment trajectories. | Black-box. |

## 3. Activation-reading methods

- **Natural Language Autoencoders** (Anthropic, 2026). These translate activations to text and back. Released checkpoints cover Qwen2.5-7B (L20), Gemma-3-12B (L32), Gemma-3-27B (L41) and Llama-3.3-70B (L53). See [`nla_paper_review.md`](nla_paper_review.md).
- **Activation Oracles** (Karvonen et al., 2512.15674; *Building Better Activation Oracles*, 2606.02609). These answer natural-language questions about activations. Known failure modes are text inversion and sensitivity to question phrasing (2607.23379).
- **LEACE** (Belrose et al., 2023). Linear concept erasure, used here to remove the format direction.

## 4. Datasets and code used or referenced

| Resource | Use in this project | License |
|---|---|---|
| [EvalAwareBench](https://github.com/aisa-group/decomposing-eval-awareness) (`aisa-group/EvalAwareBench`) | Source of the F1–F8 cue taxonomy; candidate external test set | CC BY-NC 4.0 |
| [EvalDetectBench / aware-bench](https://huggingface.co/datasets/el7982/aware-bench) (`el7982/aware-bench`) | Natural eval and deploy transcripts for the probe transfer test (`data/external/`) | CC BY-NC-SA 4.0 |
| [Needham et al. eval_awareness](https://huggingface.co/datasets/jjpn2/eval_awareness) | Candidate natural eval/deploy set | Gated; mixed source licenses |
| [steering-eval-awareness-public](https://github.com/tim-hua-01/steering-eval-awareness-public) | Reference implementation for contrastive steering | MIT |
| [natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders) | NLA inference | Apache-2.0 (checkpoints inherit base-model licenses) |
| [activation_oracles](https://github.com/adamkarvonen/activation_oracles) | Activation Oracle arm | See repository |
| [concept-erasure](https://github.com/EleutherAI/concept-erasure), [nnsight](https://github.com/ndif-team/nnsight), [SAELens](https://github.com/jbloomAus/SAELens) | Erasure, interventions, SAE features | MIT |

## 5. Positioning

Prior work has shown that eval awareness exists, that it can be decoded linearly, and that it can be steered. It has not:
1. separated **format** from **context** at the level of internal representations and tested each causally, or
2. validated **activation-based detectors** (NLAs, Activation Oracles, probes) against controlled ground truth.

This project addresses both.
