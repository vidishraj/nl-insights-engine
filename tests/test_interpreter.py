"""Interpreter: NL -> typed IR, and follow-ups as IR merge (diff, not chat replay)."""

from __future__ import annotations

from typing import Any

from nl_insights.interpreter import QueryIR, interpret, merge_ir, semantic_context
from nl_insights.provider import LLMRequest
from nl_insights.semantic import (
    ColumnBinding,
    MeasureBinding,
    ReturnsConvention,
    Role,
    SemanticModel,
)


def _model() -> SemanticModel:
    return SemanticModel(
        dataset_id="t",
        table="dataset",
        bindings=[
            ColumnBinding(
                column="stock_code", role=Role.ENTITY_KEY, entity="product", confidence=0.9
            ),
            ColumnBinding(column="country", role=Role.DIMENSION, confidence=0.9),
            ColumnBinding(column="invoice_date", role=Role.EVENT_TIME, confidence=0.9),
        ],
        returns=ReturnsConvention(kind="derived_negative_quantity"),
        measures=[
            MeasureBinding(
                name="net_revenue",
                expression="sum(quantity × rate)",
                grain="row",
                additive=True,
                available=True,
                sql="sum(q*r)",
            )
        ],
    )


class Stub:
    name = "stub"

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        return self._response


def test_semantic_context_exposes_measures_and_dimensions() -> None:
    ctx = semantic_context(_model())
    assert {m["name"] for m in ctx["measures"]} == {"net_revenue"}
    assert ctx["dimensions"] == ["country"]
    assert ctx["event_time"] == "invoice_date"


def test_semantic_context_maps_each_measure_to_its_domain_columns() -> None:
    # The generalisation fix: measure NAMES stay generic (net_revenue/units_sold), but the
    # interpreter must see the CONCRETE columns each aggregates — which carry a non-retail
    # dataset's own words — so the model can map 'freight cost' -> net_revenue without the
    # domain vocabulary being baked in.
    model = SemanticModel(
        dataset_id="freight",
        table="dataset",
        bindings=[
            ColumnBinding(column="freight_cost", role=Role.MONETARY_AMOUNT, confidence=0.95),
            ColumnBinding(column="weight_kg", role=Role.ADDITIVE_QUANTITY, confidence=0.8),
            ColumnBinding(column="rate_per_kg", role=Role.MONETARY_RATE, confidence=1.0),
            ColumnBinding(column="ship_date", role=Role.EVENT_TIME, confidence=1.0),
            ColumnBinding(column="carrier", role=Role.DIMENSION, confidence=0.9),
        ],
        returns=ReturnsConvention(kind="none"),
        measures=[
            MeasureBinding(
                name="net_revenue",
                expression="sum(monetary_amount)  [verified stored amount]",
                grain="row",
                additive=True,
                available=True,
                sql='sum("freight_cost")',
            ),
            MeasureBinding(
                name="units_sold",
                expression="sum(additive_quantity)",
                grain="row",
                additive=True,
                available=True,
                sql='sum("weight_kg")',
            ),
        ],
    )
    by_name = {m["name"]: m for m in semantic_context(model)["measures"]}
    assert by_name["net_revenue"]["columns"] == ["freight_cost"]  # the domain word is visible
    assert by_name["units_sold"]["columns"] == ["weight_kg"]


def test_interpret_parses_a_typed_plan() -> None:
    response = {
        "intent": "top 10 products by revenue",
        "measures": ["net_revenue"],
        "group_by": ["stock_code"],
        "top_k": {"measure": "net_revenue", "k": 10, "direction": "desc"},
    }
    ir = interpret(Stub(response), _model(), "top 10 products by revenue")
    assert ir.measures == ["net_revenue"]
    assert ir.group_by == ["stock_code"]
    assert ir.top_k is not None and ir.top_k.k == 10


def test_followup_merges_partial_over_previous() -> None:
    previous = QueryIR(intent="revenue by country", measures=["net_revenue"], group_by=["country"])
    merged = merge_ir(previous, {"time": {"named_period": "2011-03"}})
    assert merged.measures == ["net_revenue"]  # inherited
    assert merged.group_by == ["country"]  # inherited
    assert merged.time is not None and merged.time.named_period == "2011-03"  # overridden
    assert any("overrode time" in n for n in merged.notes)


def test_followup_that_changes_nothing_is_flagged_for_clarify() -> None:
    previous = QueryIR(measures=["net_revenue"], group_by=["country"])
    merged = merge_ir(previous, {})
    assert any("clarification" in n for n in merged.notes)


def test_unsupported_parts_are_noted_not_forced() -> None:
    response = {
        "intent": "sentiment of reviews",
        "measures": ["net_revenue"],
        "notes": ["no sentiment measure exists; returning revenue instead"],
    }
    ir = interpret(Stub(response), _model(), "what is the sentiment of reviews")
    assert any("sentiment" in n for n in ir.notes)
