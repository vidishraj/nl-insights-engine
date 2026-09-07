"""wb-wkn.45 Tier 2 — revenue from a single unverified amount (B), competing-amounts
clarify (C), and the CONTROL ARM that keeps the flagship files answering.

The whole safety of B is its disclosure and the strict C trigger. These pin: a single
unverified amount ANSWERS with a prose disclosure that names the column and says the check
could not be RUN (not that it failed); several competing amounts CLARIFY listing them and
NEVER silently sum one; and a file with a single (verified) amount or a derived revenue is
untouched — it still ANSWERS, which is the control arm made testable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb

from nl_insights.binder import VerdictKind, bind
from nl_insights.executor import execute
from nl_insights.interpreter import QueryIR
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model

_SYNTHETIC = Path(__file__).resolve().parents[1] / "assets" / "nl-insights" / "synthetic-retail.csv"


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


def _build(tmp_path: Path, header, rows, claims):  # type: ignore[no-untyped-def]
    from nl_insights.ingestion import ingest

    f = tmp_path / "d.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    result = ingest(f, data_dir=tmp_path / ".data", name="d")
    con = duckdb.connect(str(result.duckdb_path))
    return con, build_semantic_model(con, result.table, result.profile, _Stub(claims)), result.table


# --- (C) several competing amounts CLARIFY, never silently sum the cost column --------
def test_competing_amounts_clarify_and_never_silently_answer(tmp_path: Path) -> None:
    rows = []
    for i in range(200):
        q, p = (i % 5) + 1, round(2 + (i % 7) * 0.5, 2)
        cogs = round(q * p, 2)
        rows.append([f"I{i}", q, p, cogs, round(cogs * 1.05, 2), round(cogs * 0.05, 2)])
    con, model, _t = _build(
        tmp_path,
        ["invoice", "qty", "unit_price", "cogs", "total", "tax"],
        rows,
        {
            "invoice": ("transaction_key", None),
            "qty": ("additive_quantity", None),
            "unit_price": ("monetary_rate", None),
            "cogs": ("monetary_amount", None),
            "total": ("monetary_amount", None),
            "tax": ("monetary_amount", None),
        },
    )
    v = bind(model, QueryIR(measures=["net_revenue"]))
    assert v.kind is VerdictKind.CLARIFY and v.plan is None  # never summed one silently
    assert set(v.evidence["competing_amounts"]) == {"cogs", "total", "tax"}
    q = v.clarify_question or ""
    for col in ("cogs", "total", "tax"):
        assert col in q  # the candidates are named for the user to choose


# --- (B) a single unverified amount ANSWERS, disclosed as reported --------------------
def test_single_unverified_amount_answers_with_prose_disclosure(tmp_path: Path) -> None:
    rows = [[round(10 + i * 0.7, 2), i % 20, ["Mon", "Tue"][i % 2]] for i in range(200)]
    con, model, t = _build(
        tmp_path,
        ["total_bill", "tip_pct", "day"],
        rows,
        {
            "total_bill": ("monetary_amount", None),
            "tip_pct": ("ignore", None),
            "day": ("dimension", None),
        },
    )
    v = bind(model, QueryIR(measures=["net_revenue"]))
    # ANSWERABLE, not ANSWER_WITH_CAVEATS: the only disclosure here is an ASSUMPTION (how we read
    # the amount - "as reported", unverified), not a caveat that limits the answer. The prose
    # still surfaces below; the verdict class is reserved for a real limitation.
    assert v.kind is VerdictKind.ANSWERABLE
    a = execute(con, v.plan, model, v.caveats)
    assert a.rows[0]["net_revenue"] == con.execute(f"SELECT sum(total_bill) FROM {t}").fetchone()[0]
    disclosure = next(c for c in a.assumptions if "as reported" in c.lower())
    assert "total_bill" in disclosure  # names the column
    assert "could not be run" in disclosure and "different from it failing" in disclosure


# The retail claim map for the committed synthetic slice (its columns are the enriched shape).
_RETAIL_CLAIMS: dict[str, tuple[str, str | None]] = {
    "invoice_no": ("transaction_key", None),
    "invoice_date": ("event_time", None),
    "is_complete_quarter": ("flag", None),
    "stock_code": ("entity_key", "product"),
    "description": ("description", None),
    "line_type": ("dimension", None),
    "is_product_line": ("flag", None),
    "quantity": ("additive_quantity", None),
    "unit_price": ("monetary_rate", None),
    "line_revenue": ("monetary_amount", None),
    "is_return": ("flag", None),
    "customer_id": ("entity_key", "customer"),
    "country": ("dimension", None),
}


# --- CONTROL ARM: a single VERIFIED amount is untouched — still answers, no clarify ----
def test_control_arm_single_verified_amount_answers(tmp_path: Path) -> None:
    # the committed retail slice has ONE stored amount that verifies against qty×rate, so
    # revenue answers with no competing-amount clarify and no 'as reported' disclosure.
    from nl_insights.ingestion import ingest

    result = ingest(_SYNTHETIC, data_dir=tmp_path / ".data", name="syn")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, _Stub(_RETAIL_CLAIMS))
    amount_cols = [
        b.column for b in model.bindings if b.role.value == "monetary_amount" and not b.refuted
    ]
    assert len(amount_cols) == 1  # exactly one -> C cannot fire
    v = bind(model, QueryIR(measures=["net_revenue"], group_by=["country"]))
    assert (
        v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS} and v.plan is not None
    )
    a = execute(con, v.plan, model, v.caveats)
    assert a.rows and not any(
        "as reported" in c for c in a.assumptions
    )  # verified, no B disclosure
    con.close()


def test_control_arm_derived_revenue_answers(tmp_path: Path) -> None:
    # quantity + rate, no stored amount at all: revenue derives from qty×rate and answers,
    # exactly as raw UCI does — B/C never engage (no monetary_amount binding).
    rows = [[f"O{i}", (i % 5) + 1, round(1 + (i % 4) * 0.5, 2)] for i in range(120)]
    con, model, t = _build(
        tmp_path,
        ["oid", "qty", "price"],
        rows,
        {
            "oid": ("transaction_key", None),
            "qty": ("additive_quantity", None),
            "price": ("monetary_rate", None),
        },
    )
    v = bind(model, QueryIR(measures=["net_revenue"]))
    assert v.plan is not None
    a = execute(con, v.plan, model, v.caveats)
    truth = con.execute(f"SELECT sum(qty*price) FROM {t}").fetchone()[0]
    assert a.rows[0]["net_revenue"] == truth
    assert not any("as reported" in c for c in a.assumptions)
