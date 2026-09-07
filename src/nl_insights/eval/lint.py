"""Anti-hardcoding lint — makes the claim 'nothing dataset-specific' FALSIFIABLE.

The design rests on the engine never knowing anything about the development fixtures:
it infers structure from data, not from column names it was told to expect. That claim
is only worth anything if it can fail. This scanner fails the build if a development-
dataset identifier (a fixture column name, a convention literal) appears anywhere in the
engine source — i.e. anywhere a name could have been quietly hardcoded into a prompt,
a SQL string, or config.

Dataset-specific GOLDEN cases legitimately name real columns; those live under ``tests/``
and are out of scope by construction. This module (which must list the tokens to search
for) is the single allowed place the tokens appear, and it excludes itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Compound, dataset-specific identifiers from the two development fixtures. Generic
# English words that double as column names (country, quantity, description, product)
# are intentionally NOT listed: they appear legitimately in a dataset-agnostic ontology
# as illustrative examples, and flagging them would make the lint noise, not signal.
# What we forbid is the fixture's ACTUAL identifiers — the thing a shortcut would hardcode.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    # enriched fixture columns
    "invoice_no",
    "invoice_type",
    "invoice_date",
    "invoice_year",
    "invoice_quarter",
    "invoice_month",
    "invoice_dow",
    "invoice_hour",
    "is_complete_quarter",
    "stock_code",
    "line_type",
    "is_product_line",
    "is_revenue_line",
    "unit_price",
    "line_revenue",
    "is_return",
    "customer_id",
    "is_identified_customer",
    "is_country_known",
    "operator_note",
    "has_negative_price",
    "is_extreme_quantity",
    # raw UCI columns (camelCase — unambiguous identifiers)
    "InvoiceNo",
    "StockCode",
    "CustomerID",
    "UnitPrice",
    "InvoiceDate",
    # fixture data values a shortcut might special-case
    "DOTCOM",
    "CAKESTAND",
    # terms from datasets used only to DEVELOP or TEST (a freight file, the UCI year, and
    # example attributes/concepts) that must never be baked into a prompt, SQL, or config.
    # The prompt teaches the mapping pattern with placeholders; these guard against a
    # concrete term drifting back in — including onto the schema-description surface the
    # model reads. Kept out of the engine source entirely (they only ever named example data).
    "freight_cost",
    "weight_kg",
    "freight",
    "2011",
    "region",
    "profit",
    "supplier",
)

# Files/dirs under the scan root that are allowed to contain the tokens.
_SELF = "lint.py"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    token: str
    text: str


def default_root() -> Path:
    """The engine source tree to scan (``src/nl_insights``)."""
    return Path(__file__).resolve().parents[1]


# Extensions of the SHIPPED engine surface to scan. Python is the engine; the static UI
# (index.html and any JS) is also shipped and answers users, so a dataset-specific hint
# hardcoded THERE would be exactly the shortcut this lint exists to forbid — the earlier
# scope (only *.py) could not see it. Data-bearing trees (fixtures/, assets/, tests/) are
# deliberately NOT scanned: they legitimately contain the fixtures' real values.
_SCANNED_SUFFIXES: tuple[str, ...] = (".py", ".html", ".js")


def scan(root: Path | None = None) -> list[Violation]:
    """Return every occurrence of a forbidden token in the shipped engine source: the
    Python packages AND the shipped static UI under ``src/nl_insights`` (index.html/JS)."""
    root = root or default_root()
    violations: list[Violation] = []
    paths = sorted(p for p in root.rglob("*") if p.suffix in _SCANNED_SUFFIXES)
    for path in paths:
        if path.name == _SELF:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:  # pragma: no cover - unreadable file
            continue
        for n, line in enumerate(lines, start=1):
            for token in FORBIDDEN_TOKENS:
                if token in line:
                    violations.append(
                        Violation(path=str(path), line=n, token=token, text=line.strip())
                    )
    return violations


def main() -> int:
    violations = scan()
    if not violations:
        print("anti-hardcoding lint: clean — no dataset tokens in engine source.")
        return 0
    print(f"anti-hardcoding lint: {len(violations)} forbidden token(s) found:")
    for v in violations:
        print(f"  {v.path}:{v.line}: {v.token!r} in: {v.text}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
