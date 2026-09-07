"""A time grain is a GROUP BY, not a dropped grouping: "revenue by month" returns one row per
month, chronologically, while a single named period stays a whole-period total. Plus the
internal-coherence invariants that would have caught the silent-drop defect on any answer.

Runs against the committed synthetic slice (event_time = invoice_date), so no private data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.eval.suite import coherence_violations
from nl_insights.executor import execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import QueryIR
from nl_insights.interpreter.ir import Grain, TimeWindow
from nl_insights.provider import LLMRequest, ReplayCacheMiss
from nl_insights.semantic import build_semantic_model

_SYNTHETIC = Path(__file__).resolve().parents[1] / "assets" / "nl-insights" / "synthetic-retail.csv"


class _Heuristic:
    name = "replay"

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        raise ReplayCacheMiss("no fixture")


@pytest.fixture
def syn(tmp_path: Path):  # type: ignore[no-untyped-def]
    res = ingest(_SYNTHETIC, data_dir=tmp_path / ".data", name="syn")
    con = duckdb.connect(str(res.duckdb_path))
    try:
        yield con, build_semantic_model(con, res.table, res.profile, _Heuristic())
    finally:
        con.close()


def _ans(con, model, ir):  # type: ignore[no-untyped-def]
    v = bind(model, ir)
    assert v.plan is not None, v.reason
    a = execute(con, v.plan, model, v.caveats)
    # the FINALISED class from the finished answer (what every consumer reports), not the
    # provisional bind-time kind - coherence is asserted against the class actually shipped.
    return a.verdict_kind(), a


def test_by_month_groups_chronologically_not_a_single_total(syn) -> None:  # type: ignore[no-untyped-def]
    con, model = syn
    ir = QueryIR(measures=["net_revenue"], time=TimeWindow(grain=Grain.MONTH))
    kind, a = _ans(con, model, ir)
    assert len(a.rows) > 1  # one row per month, not one grand total
    assert "month" in a.columns
    labels = [r["month"] for r in a.rows]
    assert labels == sorted(labels)  # chronological, e.g. 2021-01, 2021-02, ...
    assert all(len(m) == 7 and m[4] == "-" for m in labels)  # readable YYYY-MM labels
    assert "GROUP BY" in a.sql
    assert a.filters_applied == []  # a bare grain is a grouping, not a filter (fault 3)
    assert not coherence_violations(kind, a)


def test_a_single_named_period_stays_a_whole_period_total(syn) -> None:  # type: ignore[no-untyped-def]
    con, model = syn
    tw = TimeWindow(grain=Grain.MONTH, named_period="2021-03")
    kind, a = _ans(con, model, QueryIR(measures=["net_revenue"], time=tw))
    assert len(a.rows) == 1  # a named month is one bucket: shape unchanged, not grouped
    assert "month" not in a.columns
    assert any("2021-03" in f for f in a.filters_applied)  # the bound is real and reported
    assert not coherence_violations(kind, a)


def test_a_grain_finer_than_the_named_period_groups_within_it(syn) -> None:  # type: ignore[no-untyped-def]
    con, model = syn
    tw = TimeWindow(grain=Grain.MONTH, named_period="2021-Q1")
    kind, a = _ans(con, model, QueryIR(measures=["net_revenue"], time=tw))
    assert [r["month"] for r in a.rows] == ["2021-01", "2021-02", "2021-03"]  # months within Q1
    assert not coherence_violations(kind, a)


def test_the_recorded_pre_fix_answer_is_rejected(syn) -> None:  # type: ignore[no-untyped-def]
    # The authoritative teeth test: not an input designed to trip a check, but the EXACT response
    # the running service returned for "revenue by month" before the fix, rebuilt field for field
    # from the captured JSON. A test built from the real defect cannot be satisfied by an invariant
    # that is blind to it - which two earlier drafts of these invariants were.
    con, model = syn
    _, real = _ans(con, model, QueryIR(measures=["net_revenue"], group_by=["country"]))

    # the six recorded lines: a bare month grain, no WHERE, empty caveats, only a day-first
    # assumption, a filter announced for a window that constrains nothing, and no unmet dimension.
    pre_ir = real.plan.ir.model_copy(
        update={"group_by": [], "time": TimeWindow(grain=Grain.MONTH), "unmet_dimensions": []}
    )
    pre_plan = real.plan.model_copy(
        update={"ir": pre_ir, "group_by": [], "time_group_grain": None}  # the fix not yet applied
    )
    pre = real.model_copy(
        update={
            "plan": pre_plan,
            "caveats": [],
            "assumptions": ["dates in 'event_date' read as day-first (DD/MM/YYYY)"],
            "filters_applied": ["time on event_date: {'grain': MONTH}"],
            "sql": 'SELECT sum("order_total") AS "net_revenue" FROM "dataset"',
        }
    )
    violations = coherence_violations(VerdictKind.ANSWER_WITH_CAVEATS, pre)
    joined = " | ".join(violations)
    assert "no caveat is disclosed" in joined  # INV1: caveats empty despite the day-first note
    assert "time grain" in joined and "unmet" in joined  # INV3: the month dropped, unrecorded
    assert "WHERE" in joined  # INV2: a filter announced for an empty window


def test_a_real_caveat_forces_the_caveats_class_the_mirror(syn) -> None:  # type: ignore[no-untyped-def]
    # The observed mirror defect, built from the real answer not an artificial break: a bare total
    # on a partitioned file comes back carrying a REAL partition caveat (the non-product money
    # inside the number). The finalised class MUST earn answer_with_caveats, and feeding the class
    # the bind-time view would have reported (answerable, because that caveat is born at execute)
    # MUST be flagged - the exact incoherence, pointing the opposite way to INV1.
    con, model = syn
    _, a = _ans(con, model, QueryIR(measures=["net_revenue"]))  # bare total, partition un-applied
    assert any("non-PRODUCT" in c for c in a.caveats)  # a real, execute-time caveat is present
    assert a.verdict_kind() is VerdictKind.ANSWER_WITH_CAVEATS  # finalised class earns it
    assert not coherence_violations(a.verdict_kind(), a)  # coherent with the finalised class
    assert coherence_violations(VerdictKind.ANSWERABLE, a)  # the mirror bites the bind-time class


def test_the_coherence_invariants_have_teeth(syn) -> None:  # type: ignore[no-untyped-def]
    # Beyond the recorded case: break each invariant against a real coherent answer and confirm it
    # is flagged, so the check fails the moment that class of defect reappears in any shape. The
    # base is a caveat-free answer (order_count carries no partition disclosure) so the control is
    # genuinely coherent under ANSWERABLE before each break.
    con, model = syn
    _, a = _ans(con, model, QueryIR(measures=["order_count"]))
    assert not a.caveats  # a caveat-free base
    assert not coherence_violations(VerdictKind.ANSWERABLE, a)  # the control: coherent

    # 1. a caveats verdict that keeps an ASSUMPTION but no caveat still discloses no limitation -
    #    the tightening the recorded case turned on: an assumption is not a caveat.
    stripped = a.model_copy(update={"caveats": [], "assumptions": ["dates read day-first"]})
    assert coherence_violations(VerdictKind.ANSWER_WITH_CAVEATS, stripped)
    # 2. a filter announced with no WHERE in the SQL
    faked = a.model_copy(update={"filters_applied": ["time on invoice_date"], "sql": "SELECT 1"})
    assert coherence_violations(VerdictKind.ANSWERABLE, faked)
    # 3a. a categorical grouping dropped without being recorded as unmet
    dropped_ir = a.plan.ir.model_copy(
        update={"group_by": ["country", "stock_code"], "unmet_dimensions": []}
    )
    dropped_plan = a.plan.model_copy(update={"ir": dropped_ir, "group_by": ["country"]})
    assert coherence_violations(VerdictKind.ANSWERABLE, a.model_copy(update={"plan": dropped_plan}))
    # 3b. a time grain dropped (in ir.time, never in group_by) without being recorded as unmet
    grain_ir = a.plan.ir.model_copy(
        update={"time": TimeWindow(grain=Grain.MONTH), "unmet_dimensions": []}
    )
    grain_plan = a.plan.model_copy(update={"ir": grain_ir, "time_group_grain": None})
    assert coherence_violations(VerdictKind.ANSWERABLE, a.model_copy(update={"plan": grain_plan}))
