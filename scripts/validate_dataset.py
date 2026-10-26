"""Check the task templates and the rendered dataset.

1. Schema: every task has both formats x both completeness shells, every slot has a factor and both levels,
   and every {SLOT} in a template resolves.
2. Information check: the hidden fact the model needs (recall, mold, address...) is present in the system
   prompt of every rendered item, so behavior differences can't come from missing information.
3. Separability: a bag-of-words classifier should predict format and context from the text, and each
   axis's classifier should be near chance on the OTHER axis. That shows the two factors vary independently.

Usage:
  python scripts/validate_dataset.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_dataset import SLOT_RE, build, load_tasks  # noqa: E402

errors = []


def check_schema(tasks):
    for t in tasks:
        tid = t.get("task_id", t["_path"])
        for key in ("task_id", "behavior", "system", "user"):
            if key not in t:
                errors.append(f"{tid}: missing '{key}'")
        slots = t.get("slots", {})
        for name, spec in slots.items():
            for k in ("factor", "realistic", "varied"):
                if k not in spec:
                    errors.append(f"{tid}: slot {name} missing '{k}'")
        templates = [t["system"].get(c) for c in ("full", "bare")]
        templates += [t["user"].get(f, {}).get(c) for f in ("natural", "benchmark") for c in ("full", "bare")]
        if any(x is None for x in templates):
            errors.append(f"{tid}: missing a system/user shell (need natural|benchmark x full|bare)")
            continue
        used = {m for tpl in templates for m in SLOT_RE.findall(tpl)}
        optional = {"F2_OVERLAY", "F6_PREFIX", "F8_SUFFIX"}
        for name in used - set(slots) - optional:
            errors.append(f"{tid}: template uses undefined slot {{{name}}}")
        for name in set(slots) - used:
            errors.append(f"{tid}: slot {name} defined but never used")
        if t["behavior"] != "topic_control":
            for name in optional:
                if name not in slots:
                    errors.append(f"{tid}: behavior task should define {name}")


def check_items(items):
    ids = Counter(i["id"] for i in items)
    for k, n in ids.items():
        if n > 1:
            errors.append(f"duplicate id {k}")
    for it in items:
        text = " ".join(m["content"] for m in it["messages"])
        if SLOT_RE.search(text):
            errors.append(f"{it['id']}: unfilled slot in rendered text")
        system = it["messages"][0]["content"].lower()
        keys = it["scoring"].get("honest_if_any") or it["scoring"].get("leak_if_any") or []
        if keys and not any(k.lower() in system for k in keys):
            errors.append(f"{it['id']}: hidden fact not found in system prompt")


def separability(items):
    rows = [i for i in items if i["behavior"] != "topic_control"]
    texts = [" ".join(m["content"] for m in r["messages"]) for r in rows]
    y_fmt = np.array([r["format"] == "benchmark" for r in rows], dtype=int)
    y_ctx = np.array([r["context"] == "eval" for r in rows], dtype=int)
    groups = [r["task_id"] for r in rows]
    n_splits = min(5, len(set(groups)))

    def cv_predict(y):
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=n_splits).split(texts, y, groups):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            clf = LogisticRegression(max_iter=2000)
            clf.fit(vec.fit_transform([texts[i] for i in tr]), y[tr])
            pred[te] = clf.predict(vec.transform([texts[i] for i in te]))
        return pred

    p_fmt, p_ctx = cv_predict(y_fmt), cv_predict(y_ctx)
    acc = lambda p, y: float((p == y).mean())
    return {
        "format clf -> format": acc(p_fmt, y_fmt),
        "context clf -> context": acc(p_ctx, y_ctx),
        "format clf -> context (want ~0.5)": acc(p_fmt, y_ctx),
        "context clf -> format (want ~0.5)": acc(p_ctx, y_fmt),
        "label correlation (want 0)": float(np.corrcoef(y_fmt, y_ctx)[0, 1]),
    }


def main():
    tasks = load_tasks()
    check_schema(tasks)
    main_items = build("main", tasks)
    single_items = build("single", tasks)
    check_items(main_items + single_items)

    print(f"tasks: {len(tasks)}  " + str(dict(Counter(t['behavior'] for t in tasks))))
    print(f"main grid items: {len(main_items)}  single-flip items: {len(single_items)}")
    cells = Counter((i["format"], i["context"], i["explicit"]) for i in main_items if i["behavior"] != "topic_control")
    print("items per cell (behavior tasks):")
    for k in sorted(cells):
        print(f"  {k}: {cells[k]}")
    lens = [len(" ".join(m["content"] for m in i["messages"]).split()) for i in main_items]
    print(f"prompt length (words): min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}")

    print("\nseparability (grouped by task, held-out tasks):")
    for k, v in separability(main_items).items():
        print(f"  {k}: {v:.2f}")

    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
