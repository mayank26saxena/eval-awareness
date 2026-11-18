"""Render every (item, cell) prompt of the follow-up experiment to JSONL, with a length table and checksums.

  python scripts/followup_render.py            # writes data/followup/rendered/{pilot,main}.jsonl + render_manifest.json
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import followup_cues as C
import followup_items as F

OUT = F.ROOT / "data/followup/rendered"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, M = F.load_all(), F.manifest()
    man, lens = {}, defaultdict(list)
    for split in ("pilot", "main"):
        rows = []
        for bench in ("confaide", "privacylens"):
            for tid in M[bench][split]:
                it = A[tid]
                for cell in C.CELLS:
                    r = C.render(it, cell)
                    rows.append(dict(id=f"{tid}__{cell}", task_id=tid, benchmark=bench, cell=cell, split=split,
                                     messages=r["messages"], hidden_fact=r["hidden_fact"]))
                    lens[(bench, cell)].append(sum(len(m["content"]) for m in r["messages"]))
        p = OUT / f"{split}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man[split] = dict(rows=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    man["mean_chars"] = {f"{b}/{c}": round(sum(v) / len(v)) for (b, c), v in sorted(lens.items())}
    (OUT / "render_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "mean_chars"}, indent=1))
    for b in ("confaide", "privacylens"):
        print(b, {c: man["mean_chars"][f"{b}/{c}"] for c in C.CELLS})


if __name__ == "__main__":
    main()
