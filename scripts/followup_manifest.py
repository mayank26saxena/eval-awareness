"""Write results/followup/MANIFEST.sha256: SHA-256 of every input, output, label file and script of the follow-up experiment.

  python3 scripts/followup_manifest.py           # write
  python3 scripts/followup_manifest.py --check   # verify every listed file against the manifest
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOBS = ["data/followup/**/*", "results_pod/followup/results/followup/**/*", "results/followup/**/*",
         "notes/followup_preregistration.md", "notes/followup_calibration/**/*", "notes/followup_validation/**/*",
         "scripts/followup_*", "tests/test_followup_*.py", "tools/followup_grader.html", "docs/followup/*"]
DEST = ROOT / "results/followup/MANIFEST.sha256"


def files():
    out = set()
    for g in GLOBS:
        out |= {p for p in ROOT.glob(g) if p.is_file() and p != DEST and p.name != ".DS_Store"}
    return sorted(out)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    if "--check" in sys.argv:
        bad = 0
        for line in DEST.read_text().splitlines():
            h, rel = line.split("  ", 1)
            p = ROOT / rel
            if not p.exists() or sha(p) != h:
                bad += 1
                print("MISMATCH", rel)
        print(f"{bad} mismatches")
        sys.exit(1 if bad else 0)
    lines = [f"{sha(p)}  {p.relative_to(ROOT)}" for p in files()]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} files -> {DEST.relative_to(ROOT)}")
