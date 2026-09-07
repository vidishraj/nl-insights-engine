"""wb-wkn.45 Tier 1 — verified-amount SELECTION and CONFIRMING-verifier semantics.

These pin the precondition (the '[verified stored amount]' label is only ever earned by the
column that itself verified) and change A (a column that FAILS the stored-amount identity is
UNVERIFIED, not REFUTED — the identity is confirming, not disqualifying). Inline CSVs; no
committed data files (the full foreign-shape battery lives in test_foreign_shapes).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

from nl_insights.binder import VerdictKind, bind
from nl_insights.interpreter import QueryIR
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model
from nl_insights.semantic.model import VerificationStatus


class _Stub:
    name = "stub"

    def __init__(self, claims: dict[str, tuple[str, str | None]]) -> None:
        self._claims = claims

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "claims": [
                {"column": c, "role": r, "entity": e, "confidence": 0.95, "reasons": ["llm"]}
                for c, (r, e) in self._claims.items()
            ]
        }


def _build(tmp_path: Path, header: list[str], rows: list[list[Any]], claims):  # type: ignore[no-untyped-def]
    from nl_insights.ingestion import ingest

    f = tmp_path / "d.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    result = ingest(f, data_dir=tmp_path / ".data", name="d")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, _Stub(claims))
    return con, model


def _supermarket(tmp_path: Path):  # type: ignore[no-untyped-def]
    # total = qty*rate*1.05 (tax), cogs = qty*rate exactly. Two competing money columns.
    rows = []
    for i in range(200):
        q = (i % 5) + 1
        p = round(2 + (i % 7) * 0.5, 2)
        cogs = round(q * p, 2)
        rows.append([f"INV{i}", q, p, cogs, round(q * p * 1.05, 2)])
    claims = {
        "invoice": ("transaction_key", None),
        "qty": ("additive_quantity", None),
        "unit_price": ("monetary_rate", None),
        "cogs": ("monetary_amount", None),
        "total": ("monetary_amount", None),
    }
    return _build(tmp_path, ["invoice", "qty", "unit_price", "cogs", "total"], rows, claims)


def test_verified_amount_provenance_names_the_verified_column(tmp_path: Path) -> None:
    # cogs == qty*rate verifies; total (with tax) does not. The revenue measure must sum the
    # VERIFIED column and label it as such — never earn the '[verified stored amount]' badge on
    # one column while summing another (the false-provenance bug precondition 0 closes).
    _con, model = _supermarket(tmp_path)
    rev = next(m for m in model.measures if m.name == "net_revenue")
    assert rev.sql == 'sum("cogs")'
    assert "cogs" in rev.expression and "verified stored amount" in rev.expression


def test_amount_failing_the_identity_is_unverified_not_refuted(tmp_path: Path) -> None:
    # THE change: total fails amount == qty*rate (it carries tax), but that is not positive
    # contrary evidence that total is not money — the identity simply does not apply. So total
    # is UNVERIFIED and STILL BOUND (visible to a later competing-amounts CLARIFY), never
    # refuted with confidence forced to 0.
    _con, model = _supermarket(tmp_path)
    by_col = {b.column: b for b in model.bindings}
    assert by_col["cogs"].status is VerificationStatus.VERIFIED
    assert by_col["total"].status is VerificationStatus.UNVERIFIED
    assert not by_col["total"].refuted and by_col["total"].confidence > 0
    bound_amounts = {
        b.column for b in model.bindings if b.role.value == "monetary_amount" and not b.refuted
    }
    assert bound_amounts == {"cogs", "total"}  # both remain, so competition can be seen


def test_clarify_enumerates_groupings_and_never_asserts_a_bound_column_is_absent(
    tmp_path: Path,
) -> None:
    # A bound FLAG (e.g. 'survived') is not a grouping dimension. Grouping by it must CLARIFY by
    # ENUMERATING what exists, and must not say 'this dataset has no survived column' when the
    # column is right there — a factually false statement is worse than a refusal.
    rows = [[f"P{i}", i % 2, ["A", "B", "C"][i % 3], round(1 + i * 0.5, 2)] for i in range(60)]
    claims = {
        "pid": ("transaction_key", None),
        "survived": ("flag", None),
        "klass": ("dimension", None),
        "fare": ("monetary_amount", None),
    }
    _con, model = _build(tmp_path, ["pid", "survived", "klass", "fare"], rows, claims)
    v = bind(model, QueryIR(measures=["net_revenue"], unmet_dimensions=["survived"]))
    assert v.kind is VerdictKind.CLARIFY
    q = v.clarify_question or ""
    assert "no survived column" not in q.lower()  # never assert non-existence of a bound column
    assert "klass" in q  # enumerates a real grouping
    assert v.evidence.get("bound_but_not_groupable") == ["survived"]
