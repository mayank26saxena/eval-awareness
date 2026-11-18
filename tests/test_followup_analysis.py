"""Recovery test for the follow-up analysis: simulate replies with known true effects and a judge that is wrong 15% of
the time, label 3% with the truth (standing in for Claude), and check the corrected estimates recover the true effects.
Run: .venv/bin/python -m unittest tests.test_followup_analysis -v   (about a minute)"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EFF = {"C0": 0, "C1": .10, "C2": 0, "C3": .05, "C4": .10, "C5": .15, "C6": .05, "C7": .15, "C8": .20, "C9": .35,
       "C10": .20, "C11": .0}


class Recovery(unittest.TestCase):
    def test_rogan_gladen_recovers_true_effects(self):
        rng = random.Random(0)
        d = Path(tempfile.mkdtemp())
        with open(d / "j.jsonl", "w") as fj, open(d / "c.jsonl", "w") as fc:
            for m in ("mA", "mB", "mC", "mD"):
                for b in ("confaide", "privacylens"):
                    for t in range(120):
                        base = rng.uniform(.2, .6)
                        for c, e in EFF.items():
                            for k in range(4):
                                truth = rng.random() < base + e
                                judge = truth if rng.random() < .85 else rng.random() < .6
                                fj.write(json.dumps(dict(id=f"{b}{t}__{c}", task_id=f"{b}_{t}", cell=c, benchmark=b,
                                                         model=m, k=k, good=judge)) + "\n")
                                if rng.random() < .03:
                                    fc.write(json.dumps(dict(task_id=f"{b}_{t}", cell=c, model=m, k=k, claude_good=truth)) + "\n")
        out = d / "r.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/followup_analyze.py"), "--judged", str(d / "j.jsonl"),
                        "--claude", str(d / "c.jsonl"), "--B", "200", "--out", str(out)], check=True, capture_output=True)
        r = json.loads(out.read_text())
        for name, true in (("N: C1-C0", .10), ("S: C2-C0", 0), ("E: C3-C0", .05), ("P: C8-C0", .20)):
            self.assertLess(abs(r[name]["est"] - true), .04, (name, r[name]["est"]))
            self.assertTrue(r[name]["lo"] <= true <= r[name]["hi"], (name, r[name]))


if __name__ == "__main__":
    unittest.main()
