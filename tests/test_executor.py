"""Executor: compile BoundPlan to SQL, run, and render a plan-derived explanation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.executor import Answer, execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import (
    FrequencyConstraint,
    NumericFilter,
    QueryIR,
    TimeWindow,
    TopK,
)
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model

_ENRICHED = Path(__file__).resolve().parents[1] / "assets" / "nl-insights" / "retail-enriched.csv"


class StubProvider:
    name = "stub"

    def __init__(self, claims: dict[str, tuple[str, str | None]]) -> None:
        self._claims = claims

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "claims": [
                {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                for c, (r, e) in self._claims.items()
            ]
        }


def _built(tmp_path: Path):  # type: ignore[no-untyped-def]
    lines = ["order_id,ts,product,qty,price,customer"]
    for o in range(8):
        for k in range(3):
            lines.append(f"O{o},2021-01-{o + 1:02d},P{k},{2 + k},{1.5 + k},C{o}")
    src = tmp_path / "orders.csv"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ingest(src, data_dir=tmp_path / ".data")
    con = duckdb.connect(str(result.duckdb_path))
    claims = {
        "order_id": ("transaction_key", None),
        "ts": ("event_time", None),
        "product": ("entity_key", "product"),
        "qty": ("additive_quantity", None),
        "price": ("monetary_rate", None),
        "customer": ("entity_key", "customer"),
    }
    model = build_semantic_model(con, result.table, result.profile, StubProvider(claims))
    return con, model


def _answer(con, model, ir: QueryIR) -> Answer:  # type: ignore[no-untyped-def]
    v = bind(model, ir)
    assert v.plan is not None, v.reason
    return execute(con, v.plan, model, v.caveats)


def _partitioned(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Products plus several non-product line kinds (postage/fees/adjustments) — a fact
    partition exists, so a revenue total that does NOT apply it must disclose what it
    includes, quantified."""
    lines = ["sku,description,line_kind,is_prod,qty,price"]
    for k in range(6):
        for _ in range(3):
            lines.append(f"P{k},DESC {k},PRODUCT,1,2,{k + 1}")
    for cat, price in [("POSTAGE", 1000), ("FEE", -50), ("MANUAL", -30), ("ADJUST", -20)]:
        for _ in range(3):
            lines.append(f"S_{cat},DESC {cat},{cat},0,2,{price}")
    src = tmp_path / "partitioned.csv"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ingest(src, data_dir=tmp_path / ".data")
    con = duckdb.connect(str(result.duckdb_path))
    claims = {
        "sku": ("entity_key", "product"),
        "description": ("description", None),
        "line_kind": ("dimension", None),
        "is_prod": ("flag", None),
        "qty": ("additive_quantity", None),
        "price": ("monetary_rate", None),
    }
    model = build_semantic_model(con, result.table, result.profile, StubProvider(claims))
    return con, model


def test_bare_total_quantifies_included_non_product_rows(tmp_path: Path) -> None:
    con, model = _partitioned(tmp_path)
    a = _answer(con, model, QueryIR(measures=["net_revenue"]))
    # postage = 3 lines × qty 2 × price 1000 = 6000 — the largest non-product component. The
    # disclosure is a CAVEAT (the total is contaminated by non-product money), not an assumption.
    disclosure = next(c for c in a.caveats if "non-PRODUCT" in c)
    assert "POSTAGE" in disclosure and "+6,000.00" in disclosure
    assert "restrict to line_kind = PRODUCT to exclude them" in disclosure
    assert not any("non-PRODUCT" in c for c in a.assumptions)  # not in assumptions


def test_ranking_products_does_not_disclose_non_product_rows(tmp_path: Path) -> None:
    con, model = _partitioned(tmp_path)
    # grouping by a product identifier APPLIES the partition, so there is nothing to
    # disclose — a disclosure that fires when non-product rows are already excluded is
    # its own defect.
    a = _answer(con, model, QueryIR(measures=["net_revenue"], group_by=["description"]))
    assert not any("non-PRODUCT" in c for c in [*a.assumptions, *a.caveats])


def test_no_partition_means_no_disclosure(tmp_path: Path) -> None:
    # The orders fixture has no fact partition (no line-type/flag split) → a bare total
    # must NOT invent a disclosure.
    con, model = _built(tmp_path)
    a = _answer(con, model, QueryIR(measures=["net_revenue"]))
    assert not any("non-" in c and "value(s)" in c for c in [*a.assumptions, *a.caveats])


def test_top_group_with_topk(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(
        con,
        model,
        QueryIR(
            measures=["net_revenue"], group_by=["product"], top_k=TopK(measure="net_revenue", k=2)
        ),
    )
    assert len(a.rows) == 2
    values = [r["net_revenue"] for r in a.rows]
    assert values == sorted(values, reverse=True)  # ordered by the measure
    assert "product" in a.columns
    assert a.formula and "net_revenue" in a.formula[0]
    assert "grouped by product" in a.explanation


def test_time_filter_single_value(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(
        con,
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="day", named_period="2021-01-01")),
    )
    assert len(a.rows) == 1
    assert a.rows[0]["net_revenue"] > 0
    assert "strftime" in a.sql  # the period filter compiled


def test_frequency_bought_once(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    # each customer appears in exactly one order → all 8 bought once.
    a = _answer(
        con,
        model,
        QueryIR(
            frequency=FrequencyConstraint(entity="customer", count_of="order_id", op="eq", n=1)
        ),
    )
    assert a.rows[0]["entity_count"] == 8
    assert "share" in a.rows[0]  # the matching group's share of the total is stated


def test_basket_pairs(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(con, model, QueryIR(basket=True, top_k=TopK(measure="pair_count", k=5)))
    assert {"product_a", "product_b", "pair_count"} <= set(a.columns)
    assert a.rows[0]["pair_count"] > 0


def test_answer_is_serialisable(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(con, model, QueryIR(measures=["net_revenue"], group_by=["customer"]))
    assert isinstance(Answer.model_validate_json(a.model_dump_json()), Answer)


def test_degenerate_filter_is_annotated(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(
        con,
        model,
        QueryIR(
            measures=["net_revenue"],
            group_by=["product"],
            time=TimeWindow(grain="day", named_period="1999-01-01"),
        ),  # matches nothing
    )
    assert any("empty result" in c for c in a.caveats)


def test_ungrouped_zero_match_aggregate_is_not_a_confident_zero(tmp_path: Path) -> None:
    # The dangerous shape wb-wkn.39 exposed: an UNGROUPED total over a filter that matches
    # nothing returns one row (0/None), not an empty set. It must carry the caveat, never read
    # as a real 0. (A real-column numeric filter out of range also covers the numeric audit.)
    con, model = _built(tmp_path)
    a = _answer(
        con,
        model,
        QueryIR(
            measures=["net_revenue"],
            numeric_filters=[NumericFilter(column="qty", op="gt", value=10_000)],
        ),
    )
    assert len(a.rows) == 1  # the aggregate row, not an empty set
    assert any("empty result" in c and "not a zero" in c for c in a.caveats)


def test_ungrouped_total_with_no_filter_carries_no_false_zero_match(tmp_path: Path) -> None:
    con, model = _built(tmp_path)
    a = _answer(con, model, QueryIR(measures=["net_revenue"]))
    assert a.rows[0]["net_revenue"] and not any("empty result" in c for c in a.caveats)


_ENRICHED_CLAIMS: dict[str, tuple[str, str | None]] = {
    "invoice_no": ("transaction_key", None),
    "invoice_type": ("dimension", None),
    "invoice_date": ("event_time", None),
    "invoice_year": ("dimension", None),
    "invoice_quarter": ("dimension", None),
    "invoice_month": ("dimension", None),
    "invoice_dow": ("dimension", None),
    "invoice_hour": ("dimension", None),
    "is_complete_quarter": ("flag", None),
    "stock_code": ("entity_key", "product"),
    "description": ("description", None),
    "line_type": ("dimension", None),
    "is_product_line": ("flag", None),
    "is_revenue_line": ("flag", None),
    "quantity": ("additive_quantity", None),
    "unit_price": ("monetary_rate", None),
    "line_revenue": ("monetary_amount", None),
    "is_return": ("flag", None),
    "customer_id": ("entity_key", "customer"),
    "is_identified_customer": ("flag", None),
    "country": ("dimension", None),
    "is_country_known": ("flag", None),
    "operator_note": ("ignore", None),
    "has_negative_price": ("flag", None),
    "is_extreme_quantity": ("flag", None),
}


def _enriched_model(tmp_path: Path):  # type: ignore[no-untyped-def]
    result = ingest(_ENRICHED, data_dir=tmp_path / ".data", name="enriched")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, StubProvider(_ENRICHED_CLAIMS))
    return con, model


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_top_products_excludes_non_product_lines(tmp_path: Path) -> None:
    con, model = _enriched_model(tmp_path)
    v = bind(
        model,
        QueryIR(
            measures=["net_revenue"],
            group_by=["stock_code"],
            top_k=TopK(measure="net_revenue", k=10),
        ),
    )
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    a = execute(con, v.plan, model, v.caveats)
    # the partition trap: the SQL restricts to product lines, so postage etc. can't win.
    assert "line_type" in a.sql and "PRODUCT" in a.sql
    assert any("PRODUCT" in c for c in a.caveats)
    assert len(a.rows) == 10 and a.rows[0]["net_revenue"] > 0


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_top_products_by_description_matches_ground_truth(tmp_path: Path) -> None:
    # The reviewer's exact trap: a human ranks 'top products by revenue' by the readable
    # DESCRIPTION, not the opaque code. Before the label fix this silently returned
    # DOTCOM POSTAGE 186,372.79 first. The partition must apply through the label so the
    # answer is the independently-computed product ranking.
    con, model = _enriched_model(tmp_path)
    v = bind(
        model,
        QueryIR(
            measures=["net_revenue"],
            group_by=["description"],
            top_k=TopK(measure="net_revenue", k=3),
        ),
    )
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    a = execute(con, v.plan, model, v.caveats)
    assert "line_type" in a.sql and "PRODUCT" in a.sql
    descriptions = [r["description"] for r in a.rows]
    assert "DOTCOM POSTAGE" not in descriptions  # postage is not a product
    assert descriptions[0] == "REGENCY CAKESTAND 3 TIER"
    assert round(a.rows[0]["net_revenue"], 2) == 158569.32
    assert descriptions[1] == "WHITE HANGING HEART T-LIGHT HOLDER"
    assert descriptions[2] == "PARTY BUNTING"


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_bare_total_quantifies_the_non_product_money(tmp_path: Path) -> None:
    # 'net revenue' with no product grouping legitimately includes postage — but the
    # answer must SAY, in numbers, exactly what non-product money is inside it. Ground
    # truth computed independently: POSTAGE +249,878.64, FEE -199,288.18, MANUAL -69,064.41.
    con, model = _enriched_model(tmp_path)
    for gb in ([], ["country"]):  # bare total AND non-product grouping both disclose
        v = bind(model, QueryIR(measures=["net_revenue"], group_by=gb))
        a = execute(con, v.plan, model, v.caveats)
        disclosure = next(x for x in a.caveats if "non-PRODUCT" in x)
        assert "POSTAGE +249,878.64" in disclosure
        assert "FEE -199,288.18" in disclosure
        assert "line_type = PRODUCT to exclude them" in disclosure


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_product_ranking_has_no_non_product_disclosure(tmp_path: Path) -> None:
    con, model = _enriched_model(tmp_path)
    v = bind(
        model,
        QueryIR(
            measures=["net_revenue"],
            group_by=["description"],
            top_k=TopK(measure="net_revenue", k=3),
        ),
    )
    a = execute(con, v.plan, model, v.caveats)
    assert not any("non-PRODUCT" in x for x in a.assumptions)  # already excluded
