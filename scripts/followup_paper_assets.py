"""Build the preprint's figures and number macros from the blog's exported data (never typed by hand).

  python scripts/followup_export_blog.py        # first: writes and checks docs/followup/blog/<slug>/data/figures.json
  python scripts/followup_paper_assets.py       # then: docs/followup/paper/{numbers.tex,figures/*.pdf}

Every number in the paper is a macro defined in numbers.tex; the export script has already checked each one against the
registered analysis outputs.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/followup/blog/looking-like-a-test/data"
OUT = ROOT / "docs/followup/paper"
FIG = OUT / "figures"
D = json.loads((DATA / "figures.json").read_text())
ex_js = (DATA / "examples.js").read_text()
NX = json.loads(re.search(r"window\.NAME_EXAMPLES = (.*?);\n", ex_js, re.S).group(1))
REF = json.loads(re.search(r"window\.REFUSAL = (.*?);\n", ex_js, re.S).group(1))

MODELS = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B"]
INK, INK2, MUTED, RULE = "#1d272a", "#4a5458", "#687275", "#e3e1da"
BLUE, OCHRE, TERRA, GREY = "#2563b0", "#d09017", "#c9503a", "#8a9194"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.edgecolor": "#b9bec6",
                     "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42, "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})


def s1(x):
    """Signed one-decimal number for LaTeX math mode."""
    r = round(x, 1)
    return "0.0" if r == 0 else (f"+{r:.1f}" if r > 0 else f"-{abs(r):.1f}")


def ci(v):
    return f"[{s1(v['lo'])}, {s1(v['hi'])}]"


macros = {}


def m(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = val


comp = {c["code"]: c for c in D["components"]}
oth = {c["key"]: c for c in D["other"]}
R, NR, PL = oth["renaming control: C11-C0"], oth["placeholders vs renaming: C1-C11"], oth["P placement: C10-C8"]
BB, AC = D["by_bench"], D["accounting"]
cell = lambda mod, b, c: 100 * D["cells"][mod][b][c]["sonnet"]

for code, name in (("N", "N"), ("S", "S"), ("E", "E"), ("P", "P")):
    c = comp[code]
    m(f"est{name}", f"${s1(c['est'])}$"); m(f"ci{name}", f"${ci(c)}$"); m(f"p{name}", f"{c['p_holm']:.3f}")
    m(f"excl{name}", f"${s1(c['excl']['est'])}$"); m(f"exclci{name}", f"${ci(c['excl'])}$")
for key, name in ((R, "R"), (NR, "NR"), (PL, "PL"), (oth["bundle C7-C0"], "Bundle"), (oth["additivity: C7-C1-C2-C3+2*C0"], "Add")):
    m(f"est{name}", f"${s1(key['est'])}$"); m(f"ci{name}", f"${ci(key)}$")
m("exclR", f"${s1(R['excl']['est'])}$"); m("exclciR", f"${ci(R['excl'])}$")
for k, name in (("N", "N"), ("renaming", "R"), ("vs_renaming", "NR")):
    for b, bn in (("confaide", "Conf"), ("privacylens", "Priv")):
        m(f"est{name}{bn}", f"${s1(BB[k][b]['est'])}$"); m(f"ci{name}{bn}", f"${ci(BB[k][b])}$")
for mod, mn in zip(MODELS, ("Qwen", "Llama", "Mistral", "Gemma")):
    for b, bn in (("confaide", "Conf"), ("privacylens", "Priv")):
        a, p = cell(mod, b, "C0"), cell(mod, b, "C1")
        m(f"base{mn}{bn}", f"{a:.1f}"); m(f"ph{mn}{bn}", f"{p:.1f}"); m(f"real{mn}{bn}", f"{cell(mod, b, 'C11'):.1f}")
        m(f"d{mn}{bn}", f"${s1(p - a)}$"); m(f"rel{mn}{bn}", f"${s1(100 * (p - a) / a)}$\\%")
rels = [100 * (cell(x, "confaide", "C1") - cell(x, "confaide", "C0")) / cell(x, "confaide", "C0") for x in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B")]
m("relLo", f"{min(rels):.0f}"); m("relHi", f"{max(rels):.0f}")
for b, bn in (("confaide", "Conf"), ("privacylens", "Priv")):
    for k in ("up", "same", "down"):
        m(f"{k}{bn}", str(D["shifts"][b][k]))
m("flipsConf", str(D["full_flips"]["confaide"])); m("flipsPriv", str(D["full_flips"]["privacylens"]))
rec = {(r["cell"], r["bench"]): r for r in D["recog"]}
m("recNConf", f"${s1(rec[('C1', 'confaide')]['d_test'])}$"); m("recNPriv", f"${s1(rec[('C1', 'privacylens')]['d_test'])}$")
m("recProtNConf", f"${s1(rec[('C1', 'confaide')]['d_prot'])}$"); m("recProtNPriv", f"${s1(rec[('C1', 'privacylens')]['d_prot'])}$")
cue = [rec[(c, b)] for c in ("C2", "C3") for b in ("confaide", "privacylens")]
m("recCueLo", f"{min(r['d_test'] for r in cue):.0f}"); m("recCueHi", f"{max(r['d_test'] for r in cue):.0f}")
m("recCueProtLo", f"${s1(min(r['d_prot'] for r in cue))}$"); m("recCueProtHi", f"${s1(max(r['d_prot'] for r in cue))}$")
share = D["recog_share"]
for k, mn in (("qwen2.5-7b-instruct", "Qwen"), ("llama-3.1-8b-instruct", "Llama"), ("mistral-7b-instruct-v0.3", "Mistral"), ("gemma-2-9b-it", "Gemma")):
    m(f"recShare{mn}", f"{share[k]:.0f}")
G = D["measurement"]
m("kappaShort", f"{G['short_gate']['kappa']:.2f}"); m("nHuman", str(G["short_gate"]["n"]))
for k, mn in (("claude-sonnet-5.5", "Sonnet"), ("claude-opus-5.5", "Opus"), ("automated judge (Qwen2.5-32B)", "Judge")):
    g = G["gate"][k]
    m(f"kappa{mn}", f"{g['kappa']:.2f}"); m(f"kappaci{mn}", f"[{g['ci'][0]:.2f}, {g['ci'][1]:.2f}]")
for a in AC["arms"]:
    mn = {"Main study": "Main"}[a["arm"]]
    m(f"n{mn}", f"{a['replies']:,}".replace(",", "{,}")); m(f"unclear{mn}", str(a["unclear"]))
m("nGraded", f"{AC['graded']:,}".replace(",", "{,}")); m("nUnclear", str(AC["unclear"])); m("nRecog", f"{AC['recognition']:,}".replace(",", "{,}"))
m("nIncomplete", str(AC["n_incomplete"]))
m("nGradedPaper", f"{AC['graded']:,}".replace(",", "{,}")); m("nUnclearPaper", str(AC["unclear"])); m("pctUnclearPaper", f"{100*AC['unclear']/AC['graded']:.2f}")
m("refusalText", REF["reply"].replace("&", "\\&"))
m("pText", D["p_text"].replace("&", "\\&"))


# ---------- grader agreement by naming condition (could the grader's view of "Person A" bias it?) ----------
import csv
key = json.loads((ROOT / "notes/followup_calibration/key.json").read_text())
hum = {r["row"]: r["HUMAN_label"].strip().upper() for r in csv.DictReader(open(ROOT / "notes/followup_calibration/human_graded.csv", encoding="utf-8-sig"))}
son = {}
for f in sorted((ROOT / "notes/sonnet_full/gate/labels").glob("*.txt")):
    for l in f.read_text().splitlines():
        mm = re.match(r"(R\d+)\s+(LEAKED|PROTECTED)", l)
        if mm:
            son[mm.group(1)] = mm.group(2)
PH = {"C1", "C4", "C5", "C7", "C9"}
grp = {"Ph": [], "Other": []}
for row, k in key.items():
    if hum.get(row) in ("LEAKED", "PROTECTED") and row in son:
        grp["Ph" if k["cell"] in PH else "Other"].append((hum[row] == "PROTECTED", son[row] == "PROTECTED"))
def kap(v):
    n = len(v); po = sum(a == b for a, b in v) / n; pa = sum(a for a, _ in v) / n; pb = sum(b for _, b in v) / n
    pe = pa * pb + (1 - pa) * (1 - pb); return (po - pe) / (1 - pe)
assert sum(len(v) for v in grp.values()) == G["short_gate"]["n"], "gate rows by condition do not add up to the gate"
assert abs(kap(grp["Ph"] + grp["Other"]) - G["short_gate"]["kappa"]) < 0.0015, "recomputed gate kappa differs"
for g, v in grp.items():
    m(f"gateN{g}", str(len(v))); m(f"gateKappa{g}", f"{kap(v):.2f}")
    m(f"gateLenient{g}", str(sum((not a) and b for a, b in v))); m(f"gateStrict{g}", str(sum(a and not b for a, b in v)))


# ---------- reviewer-requested checks (all from tracked result files) ----------
A_DIR = ROOT / "results/followup/analysis/sonnet"
prim = json.loads((A_DIR / "primary.json").read_text())
fam = ["N: C1-C0", "S: C2-C0", "E: C3-C0", "additivity: C7-C1-C2-C3+2*C0", "P: C8-C0"]  # the pre-registered F1 family (five tests)
ps = [prim[k]["p"] for k in fam]; order = sorted(range(5), key=lambda i: ps[i]); adj = [0.0] * 5; run = 0.0
for r_, i in enumerate(order):
    run = max(run, (5 - r_) * ps[i]); adj[i] = min(1.0, run)
m("pNfive", f"{adj[0]:.3f}"); m("pPfive", f"{adj[4]:.3f}")
ex_line = re.search(r"^  N: C1-C0: (.*)$", (A_DIR / "exclude_n.txt").read_text(), re.M).group(1)
pl_clean = [float(v) for v in re.findall(r"/priv ([+-][\d.]+)", ex_line)]
assert len(pl_clean) == 4
m("exclPrivMean", f"${s1(sum(pl_clean) / 4)}$"); m("exclPrivLo", f"${s1(min(pl_clean))}$"); m("exclPrivHi", f"${s1(max(pl_clean))}$")
for code in ("N", "S", "E", "P"):
    j = comp[code]["judge"]; m(f"judge{code}", f"${s1(j['est'])}$"); m(f"judgeci{code}", f"${ci(j)}$")
m("judgeR", f"${s1(R['judge']['est'])}$")
for mod, mn in zip(MODELS, ("Qwen", "Llama", "Mistral", "Gemma")):
    m(f"pm{mn}", f"${s1(comp['N']['per_model'][mod])}$")
recq = (A_DIR / "recog.txt").read_text()
for b, bn in (("confaide", "Conf"), ("privacylens", "Priv")):
    ln = re.search(rf"^  Llama-3.1-8B-Instruct / {b}: (.*)$", recq, re.M).group(1)
    lo_, hi_ = re.search(r"C1 \d+% \[[+-]\d+; ([+-]\d+),([+-]\d+)\]", ln).groups()
    m(f"recCiN{bn}", f"$[{lo_}, {hi_}]$")
import glob as _g
rows_ = [json.loads(l) for f_ in _g.glob(str(ROOT / "results/followup/sonnet_full/judged/main/judged_*.jsonl")) if "olmo" not in f_ for l in open(f_)]
tr = {}
for b in ("confaide", "privacylens"):
    for c in ("C0", "C1"):
        rr = [r for r in rows_ if r["benchmark"] == b and r["cell"] == c]; tr[(b, c)] = 100 * sum(r["finish"] == "length" for r in rr) / len(rr)
m("truncAll", f"{100 * sum(r['finish'] == 'length' for r in rows_) / len(rows_):.1f}")
m("truncConfPub", f"{tr[('confaide', 'C0')]:.1f}"); m("truncConfPh", f"{tr[('confaide', 'C1')]:.1f}")
m("truncPrivPub", f"{tr[('privacylens', 'C0')]:.1f}"); m("truncPrivPh", f"{tr[('privacylens', 'C1')]:.1f}")
sys.path.insert(0, str(ROOT / "scripts"))
import followup_cues as Cq, followup_items as Fq  # noqa: E402
Aq = Fq.load_all(); tids = sorted({r["task_id"] for r in rows_})
m("viewCross", str(sum(len({len(Cq.judge_view(Aq[t], c)[0]) <= 2500 for c in ("C0", "C1", "C11")}) > 1 for t in tids)))
keyq = json.loads((ROOT / "notes/followup_calibration/key.json").read_text())
assert {v["model"] for v in keyq.values()} == {"Qwen/Qwen2.5-7B-Instruct"}, "gate sample is no longer Qwen-only: update the paper text"
m("nGateSampled", str(len(keyq)))

OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
(OUT / "numbers.tex").write_text("% generated by scripts/followup_paper_assets.py from the checked blog export; do not edit\n" +
                                 "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))

# ---------- examples as tables (verbatim replies or marked excerpts; disclosing words highlighted) ----------
def tex_esc(s):
    s = s.replace("\\n", " ").replace("\u2019", "'")
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return " ".join(s.split())


def show(text, hl=None, anchor=None, width=None):
    """Escape a reply, optionally cut it to `width` characters around `anchor` (marked with [...]), and highlight `hl`."""
    t = " ".join(text.replace("\\n", " ").split()).strip().strip('"').strip()
    if width and len(t) > width:
        m_ = re.search(anchor or hl, t, re.I)
        c = m_.start() if m_ else 0
        a, b = max(0, c - width // 2), min(len(t), c + width // 2)
        a = t.rfind(" ", 0, a) + 1 if a else 0
        b = t.find(" ", b) if b < len(t) and t.find(" ", b) > 0 else len(t)
        t = ("CUTMARK " if a else "") + t[a:b] + (" CUTMARK" if b < len(t) else "")
    t = tex_esc(t).replace("CUTMARK", "[\\ldots]")
    if hl:
        t = re.sub(f"({hl})", lambda mm: "\\hl{" + mm.group(1) + "}", t, flags=re.I)
    return t


VL = {"C0": "Published names", "C11": "Realistic renaming", "C1": "Placeholders"}
def tally(e, c):
    t = e["tallies"][c]
    return f"protected {t['protected']}/{t['n']}"
def shown(e, c, k):
    return next(x for x in e["shown"] if x["cond"] == c and x["k"] == k)

ex = {x["id"]: x for x in NX}
e = ex["withhold-story"]
rows = []
for c in ("C0", "C11", "C1"):
    sh = next(x for x in e["shown"] if x["cond"] == c)
    rows.append(f"{VL[c]} & {tally(e, c)} & {show(sh['reply'], None if sh['protected'] else sh['pat'])}\\\\")
fact = tex_esc(e["fact"])
t1 = ("% generated by scripts/followup_paper_assets.py; do not edit\n\\begin{table}[!tbp]\\centering\\small\n"
      "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{1.6cm}>{\\raggedright\\arraybackslash}X@{}}\\toprule\nVersion & Graded & Reply (one sample, verbatim)\\\\\\midrule\n" +
      "\n\\midrule\n".join(rows) + "\n\\bottomrule\\end{tabularx}\n"
      f"\\caption{{The same ConfAIde story (item t3\\_001, Gemma-2-9B) under the three namings. Must not be shared: {fact}. "
      "Highlighted: where the reply gives the colleague away. ``Graded'' counts all four samples of that version.}\\label{tab:example1}\n\\end{table}\n")

e2, e3 = ex["name-only"], ex["worse"]
s2a, s2b = shown(e2, "C0", 3), shown(e2, "C1", 1)
s3a, s3b = shown(e3, "C0", 0), shown(e3, "C1", 0)
def full(model_file, task, cell, k):
    """The complete reply from the graded table (the blog export cuts long replies at 1,400 characters)."""
    for l in open(ROOT / f"results/followup/sonnet_full/judged/main/judged_{model_file}.jsonl"):
        if f'"task_id": "{task}"' in l:
            r = json.loads(l)
            if r["cell"] == cell and r["k"] == k:
                assert r["good"] is (cell == "C0"), "example grade changed"
                return r["response"]
s3a = dict(s3a, reply=full("llama-3.1-8b-instruct", "privacylens_main485", "C0", 0))
s3b = dict(s3b, reply=full("llama-3.1-8b-instruct", "privacylens_main485", "C1", 0))
rows2 = [
    f"Refusal & Llama-3.1-8B, ConfAIde t3\\_063, published names & {show(REF['reply'])} & protected (sample shown)\\\\",
    f"Name dropped, secret kept & Mistral-7B, ConfAIde t3\\_026, published names & {show(s2a['reply'], s2a['pat'], width=300)} & {tally(e2, 'C0')}\\\\",
    f" & same item, placeholders & {show(s2b['reply'], s2b['pat'], width=300)} & {tally(e2, 'C1')}\\\\",
    f"Placeholders worse & Llama-3.1-8B, PrivacyLens main485, published names & {show(s3a['reply'], None, anchor='anonymized', width=300)} & {tally(e3, 'C0')}\\\\",
    f" & same item, placeholders & {show(s3b['reply'], s3b['pat'] + '|anxiety disorder|depressive disorder|Fluoxetine', anchor='Person B', width=340)} & {tally(e3, 'C1')}\\\\",
]
t2 = ("\\begin{table}[!tbp]\\centering\\small\n\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{2.5cm}>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{1.5cm}@{}}\\toprule\n"
      "Case & Model, item, version & Reply (excerpt; [\\ldots] marks cuts) & Graded\\\\\\midrule\n" + "\n\\addlinespace\n".join(rows2) +
      "\n\\bottomrule\\end{tabularx}\n\\caption{Three cases where the binary score is harder to read. Highlighted: the disclosing words. "
      "The refusal declines a harmless request and still counts as protection; the placeholder reply in the second case drops the name but keeps "
      "the identifying detail and is correctly graded as a leak; in the third, placeholders lower protection on an agent task.}\\label{tab:cases}\n\\end{table}\n")
(OUT / "examples.tex").write_text(t1 + "\n" + t2)
if (OUT / "example1.tex").exists():
    (OUT / "example1.tex").unlink()


# ---------- figures ----------
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def whisk(ax, lo, hi, y, col):
    ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=.6, solid_capstyle="butt")
    for v in (lo, hi):
        ax.plot([v, v], [y - .12, y + .12], color=col, lw=1.2, alpha=.6)


# Figure 1: effect forest, pooled and by benchmark
groups = [("Both benchmarks (pre-specified, pooled)", {"N": comp["N"], "R": R, "NR": NR}),
          ("ConfAIde stories (exploratory)", {"N": BB["N"]["confaide"], "R": BB["renaming"]["confaide"], "NR": BB["vs_renaming"]["confaide"]}),
          ("PrivacyLens agent tasks (exploratory)", {"N": BB["N"]["privacylens"], "R": BB["renaming"]["privacylens"], "NR": BB["vs_renaming"]["privacylens"]})]
lab = {"N": ("Placeholders vs published", BLUE), "R": ("Other realistic names vs published", OCHRE), "NR": ("Placeholders vs other realistic", INK2)}
fig, ax = plt.subplots(figsize=(5.6, 3.4))
y, yt, yl = 0, [], []
for g, vals in groups:
    ax.text(-6.6, y, g.upper(), fontsize=6.8, color=MUTED, fontweight="bold", va="center")
    y -= 1
    for k in ("N", "R", "NR"):
        v = vals[k]; l, col = lab[k]
        whisk(ax, v["lo"], v["hi"], y, col); ax.plot(v["est"], y, "o", color=col, ms=5, mec="white", mew=.8)
        ax.text(v["hi"] + .7, y, s1(v["est"]).replace("-", "−"), va="center", fontsize=7.5, fontfamily="monospace")
        yt.append(y); yl.append(l); y -= 1
    y -= .4
ax.axvline(0, color="#b9bec6", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5)
ax.set_xlim(-6.6, 12); ax.set_xticks([-5, 0, 5, 10]); ax.grid(axis="x", color=RULE, lw=.6); ax.set_axisbelow(True)
ax.tick_params(length=0); ax.set_xlabel("Change in protection (percentage points), 95% interval", color=INK2)
save(fig, "effect")

# Figure 2: per-model protection, published / realistic / placeholders, both benchmarks
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.0), sharey=True)
for ax, (b, title) in zip(axs, (("confaide", "ConfAIde stories"), ("privacylens", "PrivacyLens agent tasks"))):
    for i, mod in enumerate(MODELS):
        yy = len(MODELS) - 1 - i
        a, r_, p = cell(mod, b, "C0"), cell(mod, b, "C11"), cell(mod, b, "C1")
        ax.plot([min(a, r_, p), max(a, r_, p)], [yy, yy], color="#c9cecc", lw=1.6, zorder=1)
        ax.plot(a, yy, "o", mfc="white", mec=INK2, mew=1.3, ms=6, zorder=2)
        ax.plot(r_, yy, "o", color=OCHRE, ms=5, zorder=3); ax.plot(p, yy, "o", color=BLUE, ms=5, zorder=4)
        ax.text(max(a, r_, p) + 1.5, yy, s1(p - a).replace("-", "−"), va="center", fontsize=7, fontfamily="monospace")
    ax.set_title(title, fontsize=8, loc="left", color=INK); ax.set_xlim(0, 65); ax.set_xticks([0, 20, 40, 60])
    ax.set_xticklabels(["0%", "20%", "40%", "60%"]); ax.grid(axis="x", color=RULE, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0)
axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels(MODELS[::-1])
h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="Published names"), plt.Line2D([], [], marker="o", ls="", color=OCHRE, label="Other realistic names"),
     plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Placeholders")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(.55, -.1))
save(fig, "per_model")

# Figure 3: item-level shifts
fig, ax = plt.subplots(figsize=(5.6, 1.15))
for i, (b, l) in enumerate((("confaide", "ConfAIde"), ("privacylens", "PrivacyLens"))):
    acc = 0
    for k, col, tc in (("up", BLUE, "white"), ("same", "#c9cecc", INK), ("down", TERRA, "white")):
        n = D["shifts"][b][k]; ax.barh(1 - i, n, left=acc, color=col, height=.6, edgecolor=INK2 if k == "same" else "white", lw=.6 if k == "same" else .8)
        ax.text(acc + n / 2, 1 - i, str(n), ha="center", va="center", color=tc, fontsize=7, fontfamily="monospace"); acc += n
ax.set_yticks([1, 0]); ax.set_yticklabels(["ConfAIde", "PrivacyLens"]); ax.set_xlim(0, 480); ax.set_xticks([0, 120, 240, 360, 480])
ax.tick_params(length=0); ax.set_xlabel("Item–model pairs (120 items × 4 models)", color=INK2)
h = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=INK2 if c == "#c9cecc" else c, lw=.6) for c in (BLUE, "#c9cecc", TERRA)]
ax.legend(h, ["Protected more often", "No change", "Protected less often"], ncol=3, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(.5, 1.0))
save(fig, "shifts")

# Figure 4: recognition vs behaviour (Llama)
fig, ax = plt.subplots(figsize=(4.2, 2.9))
LBL = {("C1", "confaide"): "placeholders", ("C1", "privacylens"): "placeholders", ("C2", "confaide"): "format", ("C3", "confaide"): "note", ("C2", "privacylens"): "format", ("C3", "privacylens"): "note"}
for r in D["recog"]:
    if r["cell"] == "C0":
        continue
    col = BLUE if r["bench"] == "confaide" else OCHRE
    ax.plot(r["d_test"], r["d_prot"], "o", color=col, ms=5, mec="white", mew=.6)
    if (r["cell"], r["bench"]) in LBL:
        ax.annotate(LBL[(r["cell"], r["bench"])], (r["d_test"], r["d_prot"]), xytext=(5, 6 if r["d_prot"] >= 0 else -9), textcoords="offset points", fontsize=6.3, color=INK2,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=.85))
ax.axhline(0, color="#b9bec6", lw=.8); ax.axvline(0, color="#b9bec6", lw=.8); ax.grid(color=RULE, lw=.5); ax.set_axisbelow(True)
ax.set_xlabel("Change in “looks like a test” answers (points)", color=INK2); ax.set_ylabel("Change in protection (points)", color=INK2)
ax.tick_params(length=0)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, label="ConfAIde"), plt.Line2D([], [], marker="o", ls="", color=OCHRE, label="PrivacyLens")],
          frameon=False, fontsize=7, loc="upper left")
save(fig, "recognition")
# Appendix B: verbatim cue templates (read from the cue operators) and the full per-version table
sys.path.insert(0, str(ROOT / "scripts"))
import followup_cues as FC  # noqa: E402


def tex(t):
    return t.replace("\\", "\\textbackslash{}").replace("#", "\\#").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


notes = "\n".join(f"\\item \\texttt{{{tex(n.strip())}}}" for n in FC.NEUTRAL_NOTES)
wrap = tex(FC.wrap_S({"task_id": "x"}, "<user message>")).replace("<user message>", "\\textit{<user message>}")
wrap = wrap.replace("\\#\\#\\# Case " + FC.case_number({"task_id": "x"}), "\\#\\#\\# Case \\textit{NNNN}").replace("\n", "\\\\\n")
VORDER = ["C0", "C11", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C10", "C9"]
VNAME = {"C0": "Published (baseline)", "C11": "Realistic renaming", "C1": "Placeholders", "C2": "Wrapper", "C3": "Scoring note",
         "C4": "Placeholders + wrapper", "C5": "Placeholders + note", "C6": "Wrapper + note", "C7": "All three cues",
         "C8": "Instruction (system)", "C10": "Instruction (user turn)", "C9": "All cues + instruction (system)"}
assert sorted(VORDER) == sorted(FC.CELLS) and len(VORDER) == 12
tabs = []
for b, bl in (("confaide", "ConfAIde stories"), ("privacylens", "PrivacyLens agent tasks")):
    rows = []
    for c in VORDER:
        vals = []
        for mod in MODELS:
            v, base = cell(mod, b, c), cell(mod, b, "C0")
            vals.append(f"{v:.1f}" if c == "C0" else f"{v:.1f} ({s1(v - base).replace('-', '$-$')})")
        rows.append(VNAME[c] + " & " + " & ".join(vals) + "\\\\")
    tabs.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{bl}}} (120 items)}}\\\\\\midrule\n" + "\n".join(rows))
app = (
    "% generated by scripts/followup_paper_assets.py; do not edit\n"
    "\\section{Prompt templates and full results}\\label{app:full}\n"
    "\\paragraph{Templates.} The cue operators are in \\texttt{scripts/followup\\_cues.py}; the text below is read from that file. "
    "Test-case wrapper (the case number is fixed per item):\\\\[2pt]\n"
    "\\hspace*{1.5em}\\parbox{.9\\linewidth}{\\ttfamily\\small " + wrap + "}\\\\[4pt]\n"
    "Scoring note, appended to the user turn; one of five neutral notes, fixed per item, none mentioning privacy:\n"
    "\\begin{itemize}\\setlength\\itemsep{0pt}\\small\n" + notes + "\n\\end{itemize}\n"
    "Privacy instruction, appended to the system prompt or to the end of the user turn: ``" + tex(FC.P_TEXT) + "''\n\n"
    "\\paragraph{Full results.} Table~\\ref{tab:full} gives protection for every version, model and benchmark. "
    "These are descriptive rates without intervals; the pre-specified tests and their intervals are in Section~\\ref{sec:results}.\n\n"
    "\\begin{table}[h]\\centering\\small\n\\begin{tabular}{@{}lrrrr@{}}\\toprule\n"
    "Version & " + " & ".join(MODELS) + "\\\\\\midrule\n" + "\n\\midrule\n".join(tabs) + "\n\\bottomrule\\end{tabular}\n"
    "\\caption{Protection (\\%) by prompt version, model and benchmark, graded by Sonnet; unclear replies excluded. "
    "In brackets: change from the published version, in percentage points.}\\label{tab:full}\n\\end{table}\n")
(OUT / "appendix_full.tex").write_text(app)
print(f"ok: {len(macros)} macros, 4 figures -> {OUT}")
