"""Print responses for one task across the grid, side by side, for hand-checking scores.

Usage:
  python scripts/show_pairs.py results/qwen2.5-7b-instruct/scored_pilot_main.jsonl --task ebike_battery_recall
"""

import argparse
import textwrap

from common import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("--task", default=None)
    ap.add_argument("--chars", type=int, default=500)
    args = ap.parse_args()
    rows = read_jsonl(args.scored)
    tasks = [args.task] if args.task else sorted({r["task_id"] for r in rows})
    for t in tasks:
        print("=" * 100 + f"\n{t}")
        for r in sorted((r for r in rows if r["task_id"] == t), key=lambda r: (r["format"], r["context"], r["explicit"])):
            print(f"\n--- {r['format']} / {r['context']} / explicit={r['explicit']}  ->  {r.get('label', '?')}")
            print(textwrap.indent(textwrap.shorten(r["response"].replace("\n", " "), args.chars), "    "))


if __name__ == "__main__":
    main()
