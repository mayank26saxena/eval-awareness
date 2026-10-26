# Dataset: format × context × explicit

A controlled set of prompts where the **same request** is rendered with independent switches for:

| Axis | Levels | Factors (EvalAwareBench codes) |
|---|---|---|
| **Format**: how it's written | `natural` (casual prose) / `benchmark` (formal template) | F3 register + F7 task format |
| **Context**: implicit test cues | `deploy` (all realistic) / `eval` (all varied) | F1 placeholder entities, F2 too-convenient detail, F4 bare vs full context, F6 eval scaffolding |
| **Explicit**: stated grading | `absent` / `present` | F8 "your reply will be scored…" |

Full definitions: `taxonomy.yaml`.

## Behaviors
- **honesty:** the assistant knows a damaging fact (recall, mold, accident, non-accreditation), and management would prefer it not come up. The customer asks directly. Honest means the reply discloses it.
- **pii:** the assistant can see private data (home address, medical appointment reason). An unverified person asks for it. A leak means the reply reveals it.
- **topic_control:** ordinary real-use requests whose *topic* is AI evaluation (designing chatbot tests, explaining benchmarks). Rendered in deploy context only. They check whether detectors fire on the subject of evals rather than on being in one.

The hidden fact is always present in the system prompt, so behavior changes can't come from missing information. The validator checks this.

## Files
- `tasks/*.yaml` — one template per task. Slots are keyed by factor, and shells are keyed by format × completeness (full/bare).
- `rendered/pilot_main.jsonl` — the core grid. Each behavior task has 8 items; each topic control has 2.
- `rendered/pilot_single.jsonl` — one factor flipped at a time from the realistic baseline. Use it to attribute effects to individual cues.

## Commands
```bash
.venv/bin/python scripts/render_dataset.py --mode main
.venv/bin/python scripts/render_dataset.py --mode single --out data/rendered/pilot_single.jsonl
.venv/bin/python scripts/validate_dataset.py
```

## Pilot status (2026-10-25)
- 8 tasks: 4 honesty, 2 pii, 2 topic controls. That gives 52 main-grid items and 72 single-flip items.
- Validation passes. Held-out-task bag-of-words classifiers predict format (0.96) and context (1.00), but each one is at chance on the other axis (0.54 / 0.50). Label correlation is 0, so the axes vary independently.

## v3 status (2026-10-26)
- 42 tasks: 28 honesty, 10 pii, 4 topic controls. That gives 312 main-grid items (`rendered/v3_main.jsonl`) and 456 single-flip items (`rendered/v3_single.jsonl`).
- Validation passes. Format 0.97, context 1.00; cross-axis 0.53 / 0.50; label correlation 0.
- Scoring keywords for the new tasks are chosen so they don't appear in the user prompt, so a reply that echoes the question isn't scored as honest or as a leak. `used_car_accident` (pilot) still has "accident" in the user text.

## Known issues to fix before scaling
- **Length confound:** the context=eval cells use the bare (F4) shells, so they're much shorter. A detector could pick up length. The single-flip set isolates F4. Consider length-matched controls.
- **Format bundles F3 and F7**, so register and template can't be separated.
- **The heuristic keyword scoring is rough.** Use an LLM judge for the real behavior scores.
- **Scale:** the plan calls for about 40 seed tasks. Write new tasks in the same YAML schema. An LLM can draft them, but hand-check every one.
