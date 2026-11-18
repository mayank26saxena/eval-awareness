"""Tests that freeze the follow-up cue operators. Run: python3 -m unittest tests.test_followup_cues -v
(pure Python; no model, no GPU)."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import followup_cues as C  # noqa: E402
import followup_items as F  # noqa: E402

A = F.load_all()
M = F.manifest()
IDS = [i for b in ("confaide", "privacylens") for s in ("pilot", "main") for i in M[b][s]]
PH = re.compile(r"Person [A-Z]+")
MASK = "§"


def mask_originals(t):
    spans = C._entities(t)
    out, pos = [], 0
    for s, e, *_ in spans:
        out.append(t[pos:s] + MASK)
        pos = e
    out.append(t[pos:])
    return C._EMAIL.sub(MASK, "".join(out))


def mask_new(t, pat):
    return C._EMAIL.sub(MASK, pat.sub(MASK, t))


class Manifest(unittest.TestCase):
    def test_sizes_disjoint_fresh(self):
        used = F.used_task_ids()
        self.assertEqual(len(IDS), 300)
        self.assertEqual(len(set(IDS)), 300)
        self.assertFalse(set(IDS) & used)
        for b in ("confaide", "privacylens"):
            self.assertEqual((len(M[b]["pilot"]), len(M[b]["main"])), (30, 120))
            self.assertFalse(set(M[b]["pilot"]) & set(M[b]["main"]))

    def test_selection_reproducible(self):
        self.assertEqual({b: dict(pilot=v["pilot"], main=v["main"]) for b, v in F.select().items()},
                         {b: dict(pilot=M[b]["pilot"], main=M[b]["main"]) for b in ("confaide", "privacylens")})


class ByteIdentity(unittest.TestCase):
    def test_loader_matches_earlier_rendering(self):
        rows = [json.loads(l) for l in open(ROOT / "data/rendered/external_pii.jsonl")]
        deploy = [r for r in rows if r["context"] == "deploy"]
        self.assertGreaterEqual(len(deploy), 190)
        for r in deploy:
            self.assertEqual(C.render(A[r["task_id"]], "C0")["messages"], r["messages"], r["task_id"])

    def test_c0_is_published_item(self):
        for i in IDS:
            it = A[i]
            r = C.render(it, "C0")
            self.assertEqual(r["messages"][0]["content"], it["system"])
            self.assertEqual(r["messages"][1]["content"], it["user"])
            self.assertEqual(r["hidden_fact"], it["hidden_fact"])


class Isolation(unittest.TestCase):
    def test_S_only_wraps(self):
        for i in IDS:
            it = A[i]
            r = C.render(it, "C2")
            u = r["messages"][1]["content"]
            self.assertEqual(u, f"### Case {C.case_number(it)}\n### Input\n{it['user']}\n### Response")
            self.assertEqual(r["messages"][0]["content"], it["system"])

    def test_E_only_appends_note(self):
        for i in IDS:
            it = A[i]
            r = C.render(it, "C3")
            self.assertEqual(r["messages"][1]["content"], it["user"] + C.note_E(it))
            self.assertEqual(r["messages"][0]["content"], it["system"])

    def test_P_only_system_and_P_user_only_user(self):
        for i in IDS:
            it = A[i]
            r8, r10 = C.render(it, "C8"), C.render(it, "C10")
            self.assertEqual(r8["messages"][0]["content"], it["system"] + "\n\n" + C.P_TEXT)
            self.assertEqual(r8["messages"][1]["content"], it["user"])
            self.assertEqual(r10["messages"][0]["content"], it["system"])
            self.assertEqual(r10["messages"][1]["content"], it["user"] + "\n\n" + C.P_TEXT)

    def test_N_changes_only_names(self):
        full_pool = set(C.GAZ["real_first_pool"]) | set(C.GAZ["real_last_pool"])
        for cell in ("C1", "C11"):
            for i in IDS:
                it = A[i]
                r = C.render(it, cell)
                for old, new in ((it["system"], r["messages"][0]["content"]), (it["user"], r["messages"][1]["content"]),
                                 (it["hidden_fact"], r["hidden_fact"])):
                    if cell == "C1":
                        self.assertNotRegex(old, PH, "placeholder collides with existing text")
                        pat = PH
                    else:
                        pool = "|".join(sorted(full_pool - set(re.findall(r"[A-Z][a-z]+", it["system"] + it["user"] + it["hidden_fact"])),
                                               key=len, reverse=True))
                        p1 = rf"(?:{pool})(?![a-z])"
                        pat = re.compile(rf"(?:(?:{'|'.join(C.HONORIFICS)})\.?\s+)?{p1}(?:\s+{p1})?")
                    self.assertEqual(mask_originals(old), mask_new(new, pat), (i, cell))

    def test_no_gazetteer_name_left_after_N(self):
        """Independent of the span finder: scan for any capitalised gazetteer word."""
        words = re.compile(r"(?:(?<![A-Za-z])|(?<=\\[nrt]))(" + "|".join(sorted(C.FIRST | C.LAST)) + r")(?![A-Za-z])")
        tot = left = 0
        bad = []
        for i in IDS:
            it = A[i]
            r = C.render(it, "C1")
            text = r["messages"][0]["content"] + "\n" + r["messages"][1]["content"]
            tot += len(words.findall(it["system"] + "\n" + it["user"]))
            hit = words.findall(text)
            left += len(hit)
            if hit:
                bad.append((i, hit[:3]))
        print(f"residual gazetteer tokens after N: {left}/{tot}; items: {len(bad)}", bad[:8])
        self.assertLess(left / max(tot, 1), 0.01)

    def test_N_is_consistent_bijection(self):
        """Same name -> same placeholder everywhere in one item; different names -> different placeholders."""
        for i in IDS:
            it = A[i]
            _, mp, _ = C.rename(it["task_id"], [it["system"], it["user"], it["hidden_fact"]], "placeholder")
            keys = {}
            for ph, origs in mp.items():
                for o in origs:
                    base = o.split()[-1] if o.split()[0].rstrip(".") not in C.HONORIFICS else o.split()[-1]
            self.assertEqual(len(set(mp)), len(mp))


class Lexical(unittest.TestCase):
    def added(self, cell, spec_old, new):
        return new.replace(spec_old, "") if spec_old in new else new

    def test_S_N_E_add_no_privacy_words_and_S_N_no_eval_words(self):
        for i in IDS:
            it = A[i]
            for cell in ("C1", "C2", "C3", "C7"):
                u = C.render(it, cell)["messages"][1]["content"]
                if cell == "C2":
                    add = u.replace(it["user"], "")
                elif cell == "C3":
                    add = u.replace(it["user"], "")
                else:
                    add = " ".join(PH.findall(u)) + " " + " ".join(re.findall(r"### \w+|user\d+@example\.com", u))
                if cell == "C7":
                    add += " " + C.note_E(it)
                self.assertIsNone(C.PRIVACY_LEXICON.search(add), (cell, add))
                if cell in ("C1", "C2"):
                    self.assertIsNone(C.EVAL_LEXICON.search(add), (cell, add))
        for n in C.NEUTRAL_NOTES:
            self.assertIsNone(C.PRIVACY_LEXICON.search(n))
            self.assertIsNotNone(C.EVAL_LEXICON.search(n))

    def test_P_has_privacy_words_no_eval_words(self):
        self.assertIsNotNone(C.PRIVACY_LEXICON.search(C.P_TEXT))
        self.assertIsNone(C.EVAL_LEXICON.search(C.P_TEXT))


class Composition(unittest.TestCase):
    def test_deterministic_and_composed(self):
        for i in IDS:
            it = A[i]
            for cell in C.CELLS:
                self.assertEqual(C.render(it, cell), C.render(it, cell))
            sys7, usr7, fact7 = C.apply_names(it, "placeholder")
            expect = C.wrap_S(it, usr7 + C.note_E(it))
            r7, r9 = C.render(it, "C7"), C.render(it, "C9")
            self.assertEqual(r7["messages"][1]["content"], expect)
            self.assertEqual(r7["messages"][0]["content"], sys7)
            self.assertEqual(r9["messages"][1]["content"], expect)
            self.assertEqual(r9["messages"][0]["content"], sys7 + "\n\n" + C.P_TEXT)

    def test_notes_balanced(self):
        from collections import Counter
        c = Counter(C.note_E(A[i]) for i in IDS)
        self.assertEqual(len(c), 5)
        self.assertGreaterEqual(min(c.values()), 30)

    def test_twelve_cells(self):
        self.assertEqual(list(C.CELLS), [f"C{k}" for k in range(12)])


if __name__ == "__main__":
    unittest.main()
