"""Binder: one verdict per plan, five deterministic refusal classes, evidence-carrying."""

from __future__ import annotations

from nl_insights.binder import Verdict, VerdictKind, annotate_result, bind
from nl_insights.interpreter import (
    CategoricalFilter,
    FrequencyConstraint,
    PeriodComparison,
    QueryIR,
    TopK,
)
from nl_insights.semantic import (
    CategoricalValues,
    ColumnBinding,
    EntityLabel,
    MeasureBinding,
    PartitionDimension,
    PeriodCompleteness,
    ReturnsConvention,
    Role,
    SemanticModel,
)


def _model(*, measures_available: bool = True) -> SemanticModel:
    return SemanticModel(
        dataset_id="t",
        table="dataset",
        row_count=1000,
        bindings=[
            ColumnBinding(column="invoice_no", role=Role.TRANSACTION_KEY, confidence=0.9),
            ColumnBinding(column="invoice_date", role=Role.EVENT_TIME, confidence=0.9),
            ColumnBinding(
                column="stock_code", role=Role.ENTITY_KEY, entity="product", confidence=0.9
            ),
            ColumnBinding(
                column="customer_id", role=Role.ENTITY_KEY, entity="customer", confidence=0.9
            ),
            ColumnBinding(column="country", role=Role.DIMENSION, confidence=0.9),
            ColumnBinding(column="line_type", role=Role.DIMENSION, confidence=0.9),
            ColumnBinding(column="description", role=Role.DESCRIPTION, confidence=0.9),
        ],
        returns=ReturnsConvention(kind="explicit_flag", column="is_return"),
        partition_dimensions=[PartitionDimension(column="line_type", fact_value="PRODUCT")],
        labels=[
            EntityLabel(
                column="description",
                entity_column="stock_code",
                entity="product",
                strength=1.0,
                support=100,
            )
        ],
        period_completeness=[PeriodCompleteness(flag_column="is_complete_quarter")],
        measures=[
            MeasureBinding(
                name="net_revenue",
                expression="…",
                grain="row",
                additive=True,
                available=measures_available,
                sql="sum(q*r)",
            ),
            MeasureBinding(
                name="order_count",
                expression="…",
                grain="transaction",
                additive=False,
                available=measures_available,
                sql='count(DISTINCT "invoice_no")',
            ),
            MeasureBinding(
                name="basket_cooccurrence",
                expression="…",
                grain="transaction",
                additive=False,
                available=measures_available,
                sql=None,  # a self-join, not a scalar aggregate
            ),
        ],
        coverage={"customer_id": 0.75, "country": 1.0, "stock_code": 1.0},
        categorical_values={
            # line_type is exhaustive (both values observed); country is a sample (not)
            "line_type": CategoricalValues(values=["PRODUCT", "POSTAGE"], exhaustive=True),
            "country": CategoricalValues(values=["UK", "France", "Germany"], exhaustive=False),
        },
        date_assumptions={
            "invoice_date": "dates in 'invoice_date' read as day-first (DD/MM/YYYY) — a day > 12"
        },
    )


def test_out_of_domain_is_refused_before_execution() -> None:
    v = bind(_model(), QueryIR(intent="what is the weather"))
    assert v.kind is VerdictKind.REFUSE
    assert "does not map" in (v.reason or "")


def test_missing_concept_is_refused_with_evidence() -> None:
    v = bind(_model(), QueryIR(measures=["profit"]))
    assert v.kind is VerdictKind.REFUSE
    assert "profit" in v.evidence["missing"]
    assert "net_revenue" in v.evidence["available_measures"]


def test_unbound_dimension_is_refused() -> None:
    v = bind(_model(), QueryIR(measures=["net_revenue"], group_by=["customer_age"]))
    assert v.kind is VerdictKind.REFUSE
    assert "customer_age" in v.evidence["unbound"]


def test_invented_filter_value_on_an_exhaustive_column_clarifies() -> None:
    # 'how many returns' -> filter line_type IN ('return') when the values are PRODUCT/POSTAGE.
    # The value never existed; executing it would return a confident, uncaveated 0. CLARIFY
    # naming the real values instead — never answer.
    ir = QueryIR(
        measures=["net_revenue"],
        categorical_filters=[CategoricalFilter(column="line_type", op="in", values=["return"])],
    )
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.CLARIFY and v.plan is None
    assert "return" in v.clarify_question and "PRODUCT" in v.clarify_question
    assert v.evidence["actual_values"] == ["PRODUCT", "POSTAGE"]


def test_real_filter_value_on_an_exhaustive_column_answers() -> None:
    # the load-bearing case (top-products partition): a REAL value must NOT be over-refused.
    ir = QueryIR(
        measures=["net_revenue"],
        categorical_filters=[CategoricalFilter(column="line_type", op="in", values=["PRODUCT"])],
    )
    v = bind(_model(), ir)
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}
    assert v.plan is not None


def test_casing_difference_is_not_treated_as_an_invented_value() -> None:
    # 'product' vs 'PRODUCT' is a casing difference, not an invented value — do not refuse it.
    ir = QueryIR(
        measures=["net_revenue"],
        categorical_filters=[CategoricalFilter(column="line_type", op="in", values=["product"])],
    )
    assert bind(_model(), ir).kind is not VerdictKind.CLARIFY


def test_unknown_value_on_a_sampled_column_is_not_refused() -> None:
    # country is NON-exhaustive (a sample) — a value absent from the sample may still exist,
    # so refusing would be a false refusal. Let it through (the executor's zero-match guard
    # covers a genuine no-match).
    ir = QueryIR(
        measures=["net_revenue"],
        group_by=["country"],
        categorical_filters=[CategoricalFilter(column="country", op="in", values=["Narnia"])],
    )
    assert bind(_model(), ir).kind is not VerdictKind.CLARIFY


def test_degenerate_zero_match_aggregate_is_annotated_not_a_confident_zero() -> None:
    # an UNGROUPED aggregate over no matching rows returns ONE row of zeros — it must carry the
    # 'empty result, not a zero' caveat, which the old (empty-result-set only) guard missed.
    caveats = annotate_result([{"net_revenue": 0}], [], no_rows_matched=True)
    assert caveats and caveats[0].kind == "degenerate" and "not a zero" in caveats[0].detail
    # and a real match must NOT be annotated
    assert annotate_result([{"net_revenue": 5}], [], no_rows_matched=False) == []


def test_top_products_by_revenue_applies_the_partition_default() -> None:
    ir = QueryIR(measures=["net_revenue"], group_by=["stock_code"])
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    assert v.plan is not None
    # the fact partition is applied so non-product rows don't pollute the answer.
    applied = {(f.column, tuple(f.values)) for f in v.plan.applied_filters}
    assert ("line_type", ("PRODUCT",)) in applied
    assert any(c.kind == "partition" for c in v.caveats)
    assert any(c.kind == "assumption" for c in v.caveats)  # net-of-returns named


def test_grouping_by_product_label_applies_the_partition() -> None:
    # 'top products by revenue' phrased with the human-readable description column: the
    # partition must still apply, or postage (a non-product line) wins. This is the exact
    # trap the reviewer planted — grouping by 'description' must behave like 'stock_code'.
    ir = QueryIR(measures=["net_revenue"], group_by=["description"])
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    assert v.plan is not None
    applied = {(f.column, tuple(f.values)) for f in v.plan.applied_filters}
    assert ("line_type", ("PRODUCT",)) in applied
    assert any(c.kind == "partition" for c in v.caveats)


def test_bare_and_non_product_grouping_leave_the_partition_unapplied() -> None:
    # The binder keeps the DB-free decision: a bare total / non-product grouping does NOT
    # apply the fact partition (postage IS revenue). The QUANTIFIED disclosure of what
    # non-product money is included is the executor's job (it has the data) and is tested
    # there — the binder just must not silently filter here.
    for gb in ([], ["country"]):
        v = bind(_model(), QueryIR(measures=["net_revenue"], group_by=gb))
        assert v.plan is not None
        assert not v.plan.applied_filters


def test_time_query_discloses_the_date_interpretation() -> None:
    from nl_insights.interpreter import TimeWindow

    v = bind(_model(), QueryIR(measures=["net_revenue"], time=TimeWindow(named_period="2011-03")))
    assert any(
        c.kind == "assumption" and "day-first" in c.detail for c in v.caveats
    )  # the resolver's decision is surfaced, not swallowed


def test_non_time_query_does_not_mention_dates() -> None:
    v = bind(_model(), QueryIR(measures=["net_revenue"], group_by=["stock_code"]))
    assert not any("day-first" in c.detail for c in v.caveats)


def test_rank_by_unresolvable_measure_is_refused() -> None:
    # 'revenue' is not a measure this query computes ('net_revenue' is) → the ORDER BY
    # node does not bind, so it refuses instead of compiling to a BinderException.
    ir = QueryIR(
        measures=["net_revenue"], group_by=["stock_code"], top_k=TopK(measure="revenue", k=5)
    )
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.REFUSE
    assert v.evidence["order_by"] == "revenue"
    assert "net_revenue" in v.evidence["rankable"]


def test_rank_by_a_non_computed_column_is_refused() -> None:
    # ranking by a raw column the query doesn't select would be a GROUP BY / ORDER BY
    # error at run time; the binder catches it first.
    ir = QueryIR(
        measures=["net_revenue"], group_by=["stock_code"], top_k=TopK(measure="country", k=5)
    )
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.REFUSE  # 'country' is neither a measure nor a grouping here


def test_negative_top_k_is_rejected_at_the_type_boundary() -> None:
    # a rank size < 1 is not a plan; the IR itself rejects it (no negative LIMIT reaches SQL).
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TopK(measure="net_revenue", k=-5)


def test_rank_by_a_valid_grouping_is_allowed() -> None:
    # ordering by the grouping column itself is legitimate and must still answer.
    ir = QueryIR(
        measures=["net_revenue"], group_by=["stock_code"], top_k=TopK(measure="stock_code", k=5)
    )
    v = bind(_model(), ir)
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}


def test_low_coverage_entity_attaches_quantified_caveat() -> None:
    v = bind(_model(), QueryIR(measures=["net_revenue"], group_by=["customer_id"]))
    cov = next(c for c in v.caveats if c.kind == "coverage")
    assert "750/1000" in cov.detail  # numbers from the profile, not the LLM
    assert cov.metrics["coverage"] == 0.75


def test_period_comparison_excludes_incomplete_periods() -> None:
    ir = QueryIR(
        measures=["net_revenue"],
        group_by=["country"],
        period_comparison=PeriodComparison(grain="quarter"),
    )
    v = bind(_model(), ir)
    assert v.plan is not None
    assert v.plan.require_complete_period_flag == "is_complete_quarter"
    assert any(c.kind == "period" for c in v.caveats)


def test_refuted_key_makes_order_questions_refuse() -> None:
    # measures unavailable (as when the transaction key was refuted upstream).
    v = bind(_model(measures_available=False), QueryIR(measures=["order_count"]))
    assert v.kind is VerdictKind.REFUSE
    assert "order_count" in v.evidence["missing"]


def test_followup_with_no_change_is_clarify() -> None:
    ir = QueryIR(
        measures=["net_revenue"], notes=["follow-up changed nothing — needs clarification"]
    )
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.CLARIFY
    assert v.clarify_question


def test_frequency_needs_key_and_entity() -> None:
    ir = QueryIR(frequency=FrequencyConstraint(entity="customer_id", count_of="invoice_no", n=1))
    v = bind(_model(), ir)
    # both are bound here → not refused for missing; a plan is produced.
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}


def test_frequency_resolves_role_and_entity_names_to_columns() -> None:
    # the interpreter names things by ROLE ('transaction_key') and ENTITY KIND ('customer');
    # the binder must resolve them to columns, not refuse because a role name isn't a column.
    ir = QueryIR(frequency=FrequencyConstraint(entity="customer", count_of="transaction_key", n=1))
    v = bind(_model(), ir)
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}, v.reason
    assert v.plan is not None
    # rewritten to the real columns for the executor
    assert v.plan.ir.frequency.count_of == "invoice_no"
    assert v.plan.ir.frequency.entity == "customer_id"


def test_frequency_refuses_without_a_transaction_key() -> None:
    # a dataset with no transaction key must STILL refuse (do not fix by deleting guards).
    m = SemanticModel(
        dataset_id="t",
        table="dataset",
        bindings=[
            ColumnBinding(column="cust", role=Role.ENTITY_KEY, entity="customer", confidence=0.9)
        ],
        returns=ReturnsConvention(kind="none"),
    )
    v = bind(
        m,
        QueryIR(frequency=FrequencyConstraint(entity="customer", count_of="transaction_key", n=1)),
    )
    assert v.kind is VerdictKind.REFUSE and "transaction key" in v.reason


def test_basket_measure_listed_is_not_falsely_missing() -> None:
    # basket_cooccurrence has sql=None (a self-join). Listing it as a measure must NOT be
    # flagged missing by the scalar-sql check; it binds via the availability flag.
    ir = QueryIR(
        basket=True, measures=["basket_cooccurrence"], top_k=TopK(measure="pair_count", k=5)
    )
    v = bind(_model(), ir)
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}, v.reason
    assert v.plan is not None
    # basket_cooccurrence is not a scalar BoundMeasure — the executor runs the self-join
    assert [bm.name for bm in v.plan.measures] == []


def test_unmet_concept_refuses_and_does_not_answer_a_substitute() -> None:
    # 'profit margin' has no cost data → the interpreter flags it in unmet_concepts and the
    # binder REFUSES naming it, even though a substitute measure was also emitted.
    ir = QueryIR(measures=["net_revenue"], unmet_concepts=["profit", "cost"])
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.REFUSE
    assert "profit" in v.reason and "cost" in v.reason
    assert v.plan is None  # it does NOT answer the substituted revenue measure


def test_unmet_dimension_clarifies_instead_of_returning_a_scalar() -> None:
    # 'revenue by region' — no region column. The interpreter must flag it (not drop the
    # grouping and return an ungrouped total); the binder CLARIFIES, naming the gap and
    # offering the dimensions that DO exist. It must NOT answer.
    ir = QueryIR(measures=["net_revenue"], unmet_dimensions=["region"])
    v = bind(_model(), ir)
    assert v.kind is VerdictKind.CLARIFY
    assert v.plan is None  # never a bare scalar for a requested breakdown
    assert "region" in v.clarify_question
    # the offered options are real dimensions of this dataset plus the overall total
    assert "country" in v.options and "overall total (no grouping)" in v.options
    assert v.evidence["unmet_dimensions"] == ["region"]


def test_valid_grouping_still_answers_and_is_not_over_refused() -> None:
    # the fix must not over-refuse the valid case: grouping by a real dimension answers.
    v = bind(_model(), QueryIR(measures=["net_revenue"], group_by=["country"]))
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}
    assert v.plan is not None and v.plan.group_by == ["country"]


def test_degenerate_empty_result_is_annotated() -> None:
    assert annotate_result([], ["country"])[0].kind == "degenerate"
    assert annotate_result([{"country": None}], ["country"])[0].kind == "degenerate"
    assert annotate_result([{"country": "UK"}], ["country"]) == []


def test_degenerate_ignores_group_by_columns_absent_from_the_result() -> None:
    # A basket/frequency rewrite projects product_a/product_b, NOT the IR's group_by column
    # (StockCode). An absent key must NOT be read as an all-null grouping — that would attach
    # a false "grouping carries no signal" caveat to a correct answer.
    basket_rows = [
        {"product_a": "22697", "product_b": "22698", "pair_count": 905},
        {"product_a": "22910", "product_b": "22086", "pair_count": 500},
    ]
    assert annotate_result(basket_rows, ["StockCode"]) == []


def test_degenerate_still_fires_for_a_genuinely_all_null_present_grouping() -> None:
    # The guard must survive: a column that IS in the result and is all-null still warns.
    rows = [{"country": None, "revenue": 10.0}, {"country": None, "revenue": 5.0}]
    caveats = annotate_result(rows, ["country"])
    assert len(caveats) == 1 and caveats[0].kind == "degenerate"
    assert "country" in caveats[0].detail


def test_verdict_is_serialisable() -> None:
    v = bind(_model(), QueryIR(measures=["net_revenue"]))
    assert isinstance(Verdict.model_validate_json(v.model_dump_json()), Verdict)
