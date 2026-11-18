"""Append a stage's SHA-256 checksums to the follow-up pre-registration.  python scripts/followup_freeze.py <stage> files..."""
import hashlib
import sys
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
stage, files = sys.argv[1], sys.argv[2:]
lines = [f"\n### Frozen: {stage} ({datetime.date.today()})\n", "```"]
for f in files:
    lines.append(f"{hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}  {f}")
lines.append("```\n")
p = ROOT / "notes/followup_preregistration.md"
t = p.read_text()
marker = "\n## Deviations"
i = t.index(marker)
p.write_text(t[:i] + "".join(l + "\n" for l in lines) + t[i:])
print("\n".join(lines))
