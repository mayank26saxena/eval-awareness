"""Analysis for the follow-up experiment (numpy only). Frozen before the main run; see notes/followup_preregistration.md.

Inputs
  --judged   judged replies, one or more jsonl files (output of followup_judge.py): id, task_id, cell, benchmark, model, k, good
  --claude   optional jsonl of Claude validation labels: task_id, cell, model, k, claude_good (True = protected)
Method
  item-level protection = mean over k of good; contrast = paired difference in item means.
  With Claude labels (primary, deviation 6): misclassification correction (Rogan-Gladen). The judge's sensitivity and
  specificity against Claude are estimated per (model, benchmark) from the labelled replies pooled over cells, and each
  cell's judged rate p is corrected to (p + spec - 1) / (sens + spec - 1). This assumes the judge's error rates do not
  depend on the cell, which the design supports (the judge sees the same cue-free prompt in every cell) and which is
  checked by printing error rates by cell. Sensitivity analyses: per-cell prediction-powered rectifier (--mode ppi_cell,
  assumption-free but noisy) and judge-only (--mode judge).
  Intervals: item-level cluster bootstrap (resample items within benchmark, B = 10,000 by default).
  Pooling over models: DerSimonian-Laird random effects on the per-model estimates (SE from the bootstrap).
  Holm correction within family F1; equivalence by two one-sided tests at +-3 points.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

CELL_NAMES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
F1 = {  # name -> (plus cells, minus cells), all with weight +1 / -1
    "N: C1-C0": (["C1"], ["C0"]),
    "S: C2-C0": (["C2"], ["C0"]),
    "E: C3-C0": (["C3"], ["C0"]),
    "P: C8-C0": (["C8"], ["C0"]),
}
OTHER = {
    "bundle C7-C0": (["C7"], ["C0"]),
    "N+S interaction: C4-C1-C2+C0": (["C4", "C0"], ["C1", "C2"]),
    "N+E interaction: C5-C1-C3+C0": (["C5", "C0"], ["C1", "C3"]),
    "S+E interaction: C6-C2-C3+C0": (["C6", "C0"], ["C2", "C3"]),
    "additivity: C7-C1-C2-C3+2*C0": (["C7", "C0", "C0"], ["C1", "C2", "C3"]),
    "bundle+P vs P: C9-C8": (["C9"], ["C8"]),
    "bundle+P vs bundle: C9-C7": (["C9"], ["C7"]),
    "P placement: C10-C8": (["C10"], ["C8"]),
    "renaming control: C11-C0": (["C11"], ["C0"]),
    "placeholders vs renaming: C1-C11": (["C1"], ["C11"]),
}


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    return [r for r in rows if r.get("good") is not None]


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class Table:
    """Reply-level arrays for one (model, benchmark), indexed by item."""

    def __init__(self, rows, claude):
        self.items = sorted({r["task_id"] for r in rows})
        self.ix = {t: i for i, t in enumerate(self.items)}
        n = len(self.items)
        self.judge_sum = {c: np.zeros(n) for c in CELL_NAMES}
        self.judge_cnt = {c: np.zeros(n) for c in CELL_NAMES}
        self.lab_diff_sum = np.zeros(n)  # sum of (claude - judge) over labelled replies, per item
        self.lab_cnt = np.zeros(n)
        self.lab_cell = defaultdict(lambda: np.zeros(n))
        self.conf = {k: np.zeros(n) for k in ("tp", "fn", "fp", "tn")}  # judge vs Claude (positive = protected)
        self.rect_cell = {c: np.zeros(n) for c in CELL_NAMES}
        self.lab_cell_cnt = {c: np.zeros(n) for c in CELL_NAMES}
        self.claude_sum = {c: np.zeros(n) for c in CELL_NAMES}
        self.claude_cnt = {c: np.zeros(n) for c in CELL_NAMES}
        for r in rows:
            i, c = self.ix[r["task_id"]], r["cell"]
            self.judge_sum[c][i] += float(r["good"])
            self.judge_cnt[c][i] += 1
            key = (r["task_id"], r["cell"], r["model"], r["k"])
            if key in claude:
                cg = float(claude[key])
                jg = bool(r["good"])
                self.conf[("tp" if jg else "fn") if cg else ("fp" if jg else "tn")][i] += 1
                self.rect_cell[c][i] += cg - float(jg)
                self.lab_cell_cnt[c][i] += 1
                self.lab_diff_sum[i] += cg - float(r["good"])
                self.lab_cnt[i] += 1
                self.claude_sum[c][i] += cg
                self.claude_cnt[c][i] += 1

    def cell_means(self, idx, mode):
        """Mean protection per cell over the items in idx (with repeats). mode: judge | ppi | claude."""
        out = {}
        if mode == "rg":
            tp, fn, fp, tn = (self.conf[k][idx].sum() for k in ("tp", "fn", "fp", "tn"))
            sens = tp / (tp + fn) if tp + fn else 1.0
            spec = tn / (tn + fp) if tn + fp else 1.0
            youden = max(sens + spec - 1, 0.05)
        if mode == "ppi":
            w = self.lab_cnt[idx]
            rect = self.lab_diff_sum[idx].sum() / w.sum() if w.sum() > 0 else 0.0
        for c in CELL_NAMES:
            js, jc = self.judge_sum[c][idx], self.judge_cnt[c][idx]
            if mode == "rg":
                m = js.sum() / jc.sum() if jc.sum() > 0 else float("nan")
                out[c] = (m + spec - 1) / youden
                continue
            if mode == "ppi_cell":
                m = js.sum() / jc.sum() if jc.sum() > 0 else float("nan")
                lc = self.lab_cell_cnt[c][idx].sum()
                out[c] = m + (self.rect_cell[c][idx].sum() / lc if lc else 0.0)
                continue
            if mode == "claude":
                cs, cc = self.claude_sum[c][idx], self.claude_cnt[c][idx]
                out[c] = (cs.sum() / cc.sum()) if cc.sum() > 0 else float("nan")
            else:
                m = js.sum() / jc.sum() if jc.sum() > 0 else float("nan")
                out[c] = m + (rect if mode == "ppi" else 0.0)
        return out


def contrast(means, plus, minus):
    return sum(means[c] for c in plus) - sum(means[c] for c in minus)


def boot(table, spec, mode, B, rng):
    n = len(table.items)
    est = contrast(table.cell_means(np.arange(n), mode), *spec)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = contrast(table.cell_means(idx, mode), *spec)
    return est, float(np.nanstd(vals)), np.nanpercentile(vals, [2.5, 97.5])


def dl_pool(est, se):
    est, se = np.asarray(est), np.asarray(se)
    w = 1 / np.maximum(se, 1e-9) ** 2
    mu_fe = (w * est).sum() / w.sum()
    q = (w * (est - mu_fe) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    w2 = 1 / (se ** 2 + tau2)
    mu = (w2 * est).sum() / w2.sum()
    s = math.sqrt(1 / w2.sum())
    return mu, s, tau2


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--claude", default=None)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these model ids (default: every model in the files)")
    ap.add_argument("--confirmatory-models", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["rg", "ppi_cell", "ppi", "judge", "claude"], default=None,
                    help="default: rg with --claude, judge without")
    a = ap.parse_args()
    rows = load(a.judged)
    claude = {}
    if a.claude:
        for l in open(a.claude):
            r = json.loads(l)
            claude[(r["task_id"], r["cell"], r["model"], r["k"])] = r["claude_good"]
    models = a.models or sorted({r["model"] for r in rows})
    conf = a.confirmatory_models or models
    rng = np.random.default_rng(a.seed)
    mode = a.mode or ("rg" if claude else "judge")
    results = {}
    specs = {**F1, **OTHER}
    cells = {}
    for m in models:
        for bench in ("confaide", "privacylens"):
            sub = [r for r in rows if r["model"] == m and r["benchmark"] == bench]
            if sub:
                cells[(m, bench)] = Table(sub, claude)
    print(f"mode={mode}; models={models}; strata={len(cells)}; replies={len(rows)}; labelled={sum(1 for r in rows if (r['task_id'], r['cell'], r['model'], r['k']) in claude)}")
    if claude:
        print("Judge error rates against Claude, by model x benchmark (sens = judge says protected when Claude does):")
        for (m, b), t in cells.items():
            tp, fn, fp, tn = (t.conf[k].sum() for k in ("tp", "fn", "fp", "tn"))
            print(f"  {m.split('/')[-1]}/{b}: n={int(tp + fn + fp + tn)} sens={tp / max(tp + fn, 1):.2f} spec={tn / max(tn + fp, 1):.2f}")
        print("Disagreement rate by cell (all strata pooled), to check the constant-error assumption:")
        dis = {c: [0, 0] for c in CELL_NAMES}
        for r in rows:
            k = (r["task_id"], r["cell"], r["model"], r["k"])
            if k in claude:
                dis[r["cell"]][0] += bool(claude[k]) != bool(r["good"])
                dis[r["cell"]][1] += 1
        print("  " + "  ".join(f"{c} {v[0]}/{v[1]}" for c, v in dis.items() if v[1]))
    # absolute protection per cell (judge, claude-labelled subset, ppi)
    print("\nProtection by cell (pooled mean over models and benchmarks, judge-only):")
    for c in CELL_NAMES:
        v = [t.cell_means(np.arange(len(t.items)), "judge")[c] for t in cells.values()]
        print(f"  {c}: {100 * np.nanmean(v):.1f}")
    for name, spec in specs.items():
        per = {}
        for (m, bench), t in cells.items():
            if all(t.judge_cnt[c].sum() > 0 for c in spec[0] + spec[1]):
                per[(m, bench)] = boot(t, spec, mode, a.B, rng)
        results[name] = per
    # pooled over models (within benchmark and across), cluster-level
    report = {}
    for name, per in results.items():
        pooled_all = [(m, b, e, s) for (m, b), (e, s, ci) in per.items() if m in conf]
        if not pooled_all:
            continue
        # model-level estimate = mean over benchmarks (equal weight), SE combined
        by_model = defaultdict(list)
        for m, b, e, s in pooled_all:
            by_model[m].append((e, s))
        est = [np.mean([e for e, _ in v]) for v in by_model.values()]
        se = [math.sqrt(sum(s ** 2 for _, s in v)) / len(v) for v in by_model.values()]
        mu, s, tau2 = dl_pool(est, se)
        z = mu / s if s > 0 else 0.0
        p = 2 * (1 - norm_cdf(abs(z)))
        lo, hi = mu - 1.96 * s, mu + 1.96 * s
        margin = 0.03
        tost = max(norm_cdf((-margin - mu) / s) if False else 1 - norm_cdf((mu + margin) / s), 1 - norm_cdf((margin - mu) / s)) if s > 0 else 1
        agree = sum(1 for e in est if e * mu > 0)
        report[name] = dict(est=mu, se=s, lo=lo, hi=hi, p=p, tau2=tau2, equiv=bool(lo > -margin and hi < margin),
                            n_models=len(est), agree=agree, per_model={m: float(np.mean([e for e, _ in v])) for m, v in by_model.items()})
    names = list(F1)
    ps = np.array([report[n]["p"] for n in names if n in report])
    adj = dict(zip([n for n in names if n in report], holm(ps)))
    print("\n=== Family F1 (cue components; points of protection, pooled over models; Holm-adjusted p) ===")
    for n in names:
        if n in report:
            r = report[n]
            print(f"  {n:34s} {100 * r['est']:+6.1f}  [{100 * r['lo']:+6.1f}, {100 * r['hi']:+6.1f}]  p_holm={adj[n]:.3f}  "
                  f"same sign in {r['agree']}/{r['n_models']} models  equivalent(+-3)={r['equiv']}")
    print("\n=== Other contrasts (exploratory unless named in the pre-registration) ===")
    for n in OTHER:
        if n in report:
            r = report[n]
            print(f"  {n:38s} {100 * r['est']:+6.1f}  [{100 * r['lo']:+6.1f}, {100 * r['hi']:+6.1f}]  equivalent(+-3)={r['equiv']}")
    print("\n=== Per model x benchmark (point estimates, points) ===")
    for n in list(F1) + ["bundle C7-C0", "additivity: C7-C1-C2-C3+2*C0"]:
        line = []
        for (m, b), (e, s, ci) in sorted(results[n].items()):
            line.append(f"{m.split('/')[-1][:10]}/{b[:4]} {100 * e:+.1f}")
        print(f"  {n}: " + "; ".join(line))
    if a.out:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in report.items()}, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
