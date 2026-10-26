"""Render task templates into concrete chat prompts.

Each task YAML (data/tasks/*.yaml) holds slots keyed by factor (F1, F2, F4, F6, F8) and shells keyed by
format (natural | benchmark) and F4 completeness (full | bare). This script fills them in.

Modes:
  main    format x context x explicit = 8 items per behavior task (2 per topic control). The core grid.
  single  per format, start from the all-realistic prompt and flip one factor at a time. For attributing
          effects to individual cues (F1, F2, F4, F6, F8).

Usage:
  python scripts/render_dataset.py --mode main --out data/rendered/pilot_main.jsonl
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = ROOT / "data" / "tasks"
SLOT_RE = re.compile(r"\{([A-Z0-9_]+)\}")

CONTEXT_FACTORS = ["F1", "F2", "F4", "F6"]
EXPLICIT_FACTORS = ["F8"]
ALL_SWITCHES = CONTEXT_FACTORS + EXPLICIT_FACTORS


def load_tasks(task_dir=TASK_DIR):
    tasks = []
    for path in sorted(task_dir.glob("*.yaml")):
        task = yaml.safe_load(path.read_text())
        task["_path"] = str(path.relative_to(ROOT))
        tasks.append(task)
    return tasks


def fill(template, values):
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise KeyError(f"unfilled slot {{{key}}}")
        return values[key]
    return SLOT_RE.sub(sub, template)


def slot_values(task, levels):
    """levels: factor -> 'realistic' | 'varied'. Factors a task doesn't use default to realistic."""
    values = {}
    for name, spec in task.get("slots", {}).items():
        values[name] = spec[levels.get(spec["factor"], "realistic")]
    # Optional scaffolding slots that a task may omit entirely.
    for name in ("F2_OVERLAY", "F6_PREFIX", "F8_SUFFIX"):
        values.setdefault(name, "")
    return values


def render(task, fmt, levels):
    values = slot_values(task, levels)
    completeness = "bare" if levels.get("F4") == "varied" else "full"
    system = fill(task["system"][completeness], values)
    user = fill(task["user"][fmt][completeness], values)
    n_context_cues = sum(levels.get(f) == "varied" for f in CONTEXT_FACTORS)
    return {
        "task_id": task["task_id"],
        "behavior": task["behavior"],
        "format": fmt,
        "factors": {f: levels.get(f, "realistic") for f in ALL_SWITCHES},
        "n_context_cues": n_context_cues,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "scoring": task.get("scoring", {}),
        "source": task["_path"],
    }


def main_grid(task):
    contexts = task.get("context_levels", ["deploy", "eval"])
    explicits = task.get("explicit_levels", ["absent", "present"])
    for fmt, ctx, exp in itertools.product(["natural", "benchmark"], contexts, explicits):
        ctx_level = "varied" if ctx == "eval" else "realistic"
        levels = {f: ctx_level for f in CONTEXT_FACTORS}
        levels["F8"] = "varied" if exp == "present" else "realistic"
        item = render(task, fmt, levels)
        item.update(context=ctx, explicit=exp)
        item["id"] = f"{task['task_id']}__{fmt}__{ctx}__{exp}"
        yield item


def single_flips(task):
    if task["behavior"] == "topic_control":
        return
    for fmt in ["natural", "benchmark"]:
        for flip in [None] + ALL_SWITCHES:
            levels = {f: "realistic" for f in ALL_SWITCHES}
            if flip:
                levels[flip] = "varied"
            item = render(task, fmt, levels)
            item.update(context="deploy" if flip in (None, "F8") else f"only_{flip}",
                        explicit="present" if flip == "F8" else "absent")
            item["id"] = f"{task['task_id']}__{fmt}__flip_{flip or 'none'}"
            yield item


def build(mode, tasks=None):
    tasks = tasks if tasks is not None else load_tasks()
    gen = main_grid if mode == "main" else single_flips
    return [item for task in tasks for item in gen(task)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "single"], default="main")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rendered" / "pilot_main.jsonl")
    args = ap.parse_args()
    items = build(args.mode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
