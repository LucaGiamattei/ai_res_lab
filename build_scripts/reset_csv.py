"""Reset shared/evidence_template.csv to its pristine empty-with-example state."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "shared" / "evidence_template.csv"
PRISTINE = """\
challenge,system,metric,threshold,observed,status,mitigation,notes,timestamp
# example,GermanCredit-LogReg,Demographic parity difference,<= 0.10,0.187,fail,none,Modello favorisce gruppo A; gap oltre soglia regolatoria,2026-04-28T10:00:00Z
"""

if __name__ == "__main__":
    CSV.write_text(PRISTINE)
    print(f"Reset {CSV}")
