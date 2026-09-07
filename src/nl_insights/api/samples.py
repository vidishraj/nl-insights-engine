"""The bundled sample datasets and the fixed allowlist that serves them.

Serving a file and ingesting a local file are both path-traversal surfaces. There is exactly
one defence here and it is structural: an opaque id maps to a spec through this literal dict,
or it does not exist. Nothing from the request is ever joined to a base, normalised, or
prefix-checked. The path is a code constant built from a package-relative root. This is a rule
about what we OWN, not a rule about what a bad path looks like, which is the only kind of rule
that holds for inputs nobody has written yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Package-relative so it resolves at the path the SERVED process runs from (the box is deployed
# by extracting the archived tree and running from source), not only in a dev checkout.
_SAMPLES_DIR = Path(__file__).resolve().parents[3] / "assets" / "samples"


@dataclass(frozen=True)
class SampleSpec:
    id: str
    filename: str
    label: str
    description: str

    @property
    def path(self) -> Path:
        return _SAMPLES_DIR / self.filename


# The three tell the taxonomy the design argues, not three of a kind: a verified stored amount,
# a revenue that must be derived, and a file with no money at all.
SAMPLES: dict[str, SampleSpec] = {
    s.id: s
    for s in (
        SampleSpec(
            "catering",
            "catering-orders-2024.csv",
            "Catering orders (2024)",
            "202 orders whose stored total equals guests times the price per guest, so the "
            "stored-amount identity verifies. Day-first dates, and volume rising through the "
            "year so a growth question has a real answer.",
        ),
        SampleSpec(
            "storefront",
            "storefront-line-items.csv",
            "Storefront line items",
            "Sale lines with no total column, so revenue must be derived from units times a "
            "per-line price. This is the measure algebra doing real work.",
        ),
        SampleSpec(
            "sensors",
            "sensor-readings.csv",
            "Weather sensor readings",
            "Daily measurements with no money anywhere. Count and average answer honestly, and "
            "a revenue question refuses because the concept is not in the file.",
        ),
    )
}


def get_sample(sample_id: str) -> SampleSpec | None:
    """Resolve an opaque id to its spec, or None. A dict miss is a 404, never a lookup."""
    return SAMPLES.get(sample_id)
