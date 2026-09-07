"""Temporal typing: an event_time must be a real timestamp, not merely present.

The binder used to presence-check the role and ignore its storage type, so a VARCHAR
date bound as event_time and a time query then either returned a silent NULL (a
confident WRONG 'no revenue in March 2011') or threw at run time. These pin both modes:
a non-parsing text date is REFUTED so time questions refuse cleanly, and a real date (or
a text date that parses, via a recorded CAST) answers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.executor import execute
from nl_insights.executor.execute import _event_time_sql
from nl_insights.ingestion import ingest
from nl_insights.interpreter import PeriodComparison, QueryIR, TimeWindow
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model
from nl_insights.semantic.model import ColumnBinding, SemanticModel
from nl_insights.semantic.ontology import Role

_RAW_UCI = Path(__file__).resolve().parents[1] / "assets" / "raw-uci-online-retail.csv"

_CLAIMS = {
    "oid": ("transaction_key", None),
    "when": ("event_time", None),
    "item": ("entity_key", "product"),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
}


class StubProvider:
    name = "stub"

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "claims": [
                {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                for c, (r, e) in _CLAIMS.items()
            ]
        }


def _build(tmp_path: Path, date_values: list[str]):  # type: ignore[no-untyped-def]
    rows = ["oid,when,item,qty,price"]
    for i, d in enumerate(date_values):
        rows.append(f"O{i},{d},P{i % 3},{2 + i % 4},{1.5 + i % 3}")
    src = tmp_path / "t.csv"
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = ingest(src, data_dir=tmp_path / ".data", name="t")
    con = duckdb.connect(str(result.duckdb_path))
    # pass the resolved date orders, exactly like the server — so a text date is probed
    # with the RIGHT parser rather than a bare TRY_CAST
    model = build_semantic_model(
        con, result.table, result.profile, StubProvider(), date_orders=result.date_orders
    )
    return con, model


def test_event_time_sql_emits_the_recorded_expression() -> None:
    binding = ColumnBinding(column="when", role=Role.EVENT_TIME, confidence=1.0)
    base = dict(dataset_id="d", table="t", bindings=[binding], returns={"kind": "none"})
    plain = SemanticModel(**base)  # type: ignore[arg-type]
    parsed = SemanticModel(  # type: ignore[arg-type]
        **base, temporal_cast_sql={"when": "try_strptime(\"when\", ['%m/%d/%Y'])"}
    )
    assert _event_time_sql(binding, plain) == '"when"'  # native → bare column
    assert _event_time_sql(binding, parsed) == "try_strptime(\"when\", ['%m/%d/%Y'])"
    assert _event_time_sql(None, parsed) is None


def test_genuinely_non_temporal_text_is_refuted(tmp_path: Path) -> None:
    # a column that is NOT a date at all — no order resolves and TRY_CAST fails, so it is
    # REFUTED and time questions refuse cleanly (the df2566f behaviour must stay green).
    labels = ["pending", "shipped", "n/a", "unknown", "backorder", "cancelled"]
    con, model = _build(tmp_path, labels * 2)
    assert model.first_in_role(Role.EVENT_TIME) is None  # refuted

    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="month", named_period="2011-03")),
    )
    assert v.kind is VerdictKind.REFUSE and "event_time" in str(v.evidence)
    v2 = bind(
        model,
        QueryIR(
            measures=["net_revenue"],
            group_by=["item"],
            period_comparison=PeriodComparison(grain="quarter"),
        ),
    )
    assert v2.kind is VerdictKind.REFUSE  # not a BinderException
    v3 = bind(model, QueryIR(measures=["net_revenue"], group_by=["item"]))
    a3 = execute(con, v3.plan, model, v3.caveats)
    assert a3.rows and a3.rows[0]["net_revenue"] is not None  # control unaffected


def test_month_first_text_date_parses_via_strptime_not_try_cast(tmp_path: Path) -> None:
    # the precise gap: a text date 'MM/DD/YYYY HH:MM' that plain TRY_CAST cannot parse but
    # the DETECTED month-first order can (days > 12 force month-first). It must BIND and
    # ANSWER, carrying the month-first assumption — not refuse.
    con, model = _build(tmp_path, [f"03/{13 + i}/2011 10:00" for i in range(6)])
    assert model.first_in_role(Role.EVENT_TIME) is not None, "month-first text date must bind"
    assert "strptime" in model.temporal_cast_sql["when"]  # recorded parse, not TRY_CAST
    cast_ok = con.execute(
        'SELECT count(*) FILTER (WHERE TRY_CAST("when" AS TIMESTAMP) IS NOT NULL) FROM dataset'
    ).fetchone()[0]
    assert cast_ok == 0  # plain TRY_CAST parses nothing — why the old probe wrongly refuted it

    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="month", named_period="2011-03")),
    )
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    a = execute(con, v.plan, model, v.caveats)
    assert a.rows and a.rows[0]["net_revenue"] is not None  # a REAL number
    assert any("month-first" in x for x in a.assumptions)  # the interpretation is disclosed


def test_native_date_event_time_answers_time_queries(tmp_path: Path) -> None:
    con, model = _build(tmp_path, [f"2011-03-{(i % 27) + 1:02d}" for i in range(12)])
    assert model.first_in_role(Role.EVENT_TIME) is not None  # a native DATE verifies
    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="month", named_period="2011-03")),
    )
    assert v.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}
    a = execute(con, v.plan, model, v.caveats)
    assert a.rows and a.rows[0]["net_revenue"] is not None  # a REAL number, not a silent NULL


@pytest.mark.skipif(not _RAW_UCI.exists(), reason="fetch raw UCI to run")
def test_raw_uci_march_2011_answers_via_detected_order(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_eval_fixtures import _RAW_CLAIMS
    from test_eval_fixtures import StubProvider as RawStub

    result = ingest(_RAW_UCI, data_dir=tmp_path / ".data", name="raw")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(
        con, result.table, result.profile, RawStub(_RAW_CLAIMS), date_orders=result.date_orders
    )
    # InvoiceDate is VARCHAR 'MM/DD/YYYY HH:MM' — 0% via TRY_CAST, 100% via the detected
    # month-first order — so event_time BINDS and the month answers (not None, not refuse).
    assert model.first_in_role(Role.EVENT_TIME) is not None
    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="month", named_period="2011-03")),
    )
    assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
    a = execute(con, v.plan, model, v.caveats)
    assert round(a.rows[0]["net_revenue"], 2) == 683267.08  # independently computed ground truth
    assert any("month-first" in x for x in a.assumptions)


def test_time_window_schema_states_the_exclusive_end_contract() -> None:
    # the contract must live in the schema the model SEES, not only a Python comment —
    # this is what stops the model emitting an inclusive end / start==end for a month.
    from nl_insights.interpreter import QueryIR

    tw = QueryIR.model_json_schema()["$defs"]["TimeWindow"]["properties"]
    assert "EXCLUSIVE" in tw["end"]["description"]
    assert "named_period" in tw["end"]["description"]  # points the model at the right field
    assert "PREFER" in tw["named_period"]["description"]


def test_time_window_end_is_exclusive_at_the_boundary_row(tmp_path: Path) -> None:
    # The BEHAVIOUR the schema promises: a row dated exactly at `end` is EXCLUDED. Asserting
    # the boundary ROW (not just the description string) is what catches `<` mutating to
    # `<=` — the mutant that survived because only the schema wording was checked.
    con, model = _build(tmp_path, ["2021-01-01", "2021-01-02", "2021-01-03"])
    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(start="2021-01-01", end="2021-01-03")),
    )
    a = execute(con, v.plan, model, v.caveats)
    got = a.rows[0]["net_revenue"]
    lo = "\"when\" >= '2021-01-01'"
    exclusive = con.execute(
        f"SELECT sum(qty*price) FROM dataset WHERE {lo} AND \"when\" < '2021-01-03'"
    ).fetchone()[0]
    inclusive = con.execute(
        f"SELECT sum(qty*price) FROM dataset WHERE {lo} AND \"when\" <= '2021-01-03'"
    ).fetchone()[0]
    assert exclusive != inclusive  # the boundary row exists and carries revenue
    assert got == pytest.approx(exclusive)  # engine excludes it — fails if end became inclusive
    con.close()


def test_named_period_month_computes_the_full_month_with_a_window_caveat(tmp_path: Path) -> None:
    # a whole named month via named_period must cover the WHOLE month (no off-by-one) and
    # the answer must state the window it used.
    con, model = _build(tmp_path, [f"03/{13 + i}/2011 10:00" for i in range(6)])
    full = con.execute("SELECT round(sum(qty*price),2) FROM dataset").fetchone()[0]
    v = bind(
        model,
        QueryIR(measures=["net_revenue"], time=TimeWindow(grain="month", named_period="2011-03")),
    )
    a = execute(con, v.plan, model, v.caveats)
    assert round(a.rows[0]["net_revenue"], 2) == full  # the whole month, not month minus a day
    assert any("time window: 2011-03" in c for c in a.caveats)  # the window is stated


def test_last_n_compiles_a_real_where_and_filters_on_a_long_dataset(tmp_path: Path) -> None:
    # A dataset spanning FIVE years: 'last 2 years' must emit a WHERE and return LESS than the
    # whole-table total (the original bug hid behind a <1-year dataset where they coincided).
    dates = [f"{y}-{m:02d}-15" for y in range(2019, 2024) for m in range(1, 13)]
    con, model = _build(tmp_path, dates)
    total = con.execute("SELECT sum(qty*price) FROM dataset").fetchone()[0]
    v = bind(model, QueryIR(measures=["net_revenue"], time=TimeWindow(grain="year", last_n=2)))
    a = execute(con, v.plan, model, v.caveats)
    assert "WHERE" in a.sql  # a claimed window MUST appear in the SQL
    assert "max(" in a.sql  # anchored on the data's latest date, not today
    assert a.rows[0]["net_revenue"] < total  # genuinely filtered, not the whole table
    # the anchor (latest date in the data) is stated, not left as a silent assumption
    assert any("relative to the latest date in the data" in c for c in a.caveats)


def test_last_n_over_a_short_dataset_returns_all_but_discloses_it(tmp_path: Path) -> None:
    # The original bug's shape: data under a year, 'last 2 years' covers everything. The number
    # equals the total (correct), but the answer must SAY the window covers the whole dataset —
    # and the SQL must still carry the WHERE, so no caveat claims a filter absent from the SQL.
    dates = [f"2011-{m:02d}-15" for m in range(1, 7)]  # six months
    con, model = _build(tmp_path, dates)
    total = con.execute("SELECT sum(qty*price) FROM dataset").fetchone()[0]
    v = bind(model, QueryIR(measures=["net_revenue"], time=TimeWindow(grain="year", last_n=2)))
    a = execute(con, v.plan, model, v.caveats)
    assert "WHERE" in a.sql
    assert abs(a.rows[0]["net_revenue"] - total) < 1e-6  # coincidentally the whole span
    assert any("this is the whole dataset" in c for c in a.caveats)


def test_no_window_caveat_ever_claims_a_filter_absent_from_the_sql(tmp_path: Path) -> None:
    # The structural guard: for every window-bearing time field, a claimed window (a 'window'
    # caveat) must coincide with a real time predicate in the emitted SQL. last_n was the field
    # that broke this contract; assert it holds across the window shapes.
    con, model = _build(tmp_path, [f"2011-{m:02d}-15" for m in range(1, 13)])
    windows = [
        TimeWindow(grain="year", last_n=1),
        TimeWindow(grain="month", named_period="2011-03"),
        TimeWindow(start="2011-03-01", end="2011-06-01"),
    ]
    for win in windows:
        v = bind(model, QueryIR(measures=["net_revenue"], time=win))
        a = execute(con, v.plan, model, v.caveats)
        claims_window = any("time window" in c or "latest date in the data" in c for c in a.caveats)
        assert claims_window and "WHERE" in a.sql, f"caveat/SQL mismatch for {win!r}"


def test_window_claim_guard_is_an_unconditional_raise_not_a_stripped_assert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guard must FAIL LOUD (a real exception, survives -O) if a window caveat ever stands
    # without a compiled clause — never a silent wrong answer. Simulate the divergence by
    # forcing the clause check to see no clause while a window caveat is present.
    import sys

    from nl_insights.binder.verdict import Caveat
    from nl_insights.executor.execute import WindowClaimError

    con, model = _build(tmp_path, [f"2011-{m:02d}-15" for m in range(1, 13)])
    v = bind(model, QueryIR(measures=["net_revenue"], time=TimeWindow(grain="year", last_n=1)))
    # The executor PACKAGE re-exports the function `execute`, which shadows the submodule name,
    # so both an aliased import and monkeypatch's dotted resolution land on the function. Reach
    # the module by its sys.modules key and force the clause check to see no time clause.
    module = sys.modules["nl_insights.executor.execute"]
    monkeypatch.setattr(module, "_time", lambda *a, **k: [])
    with pytest.raises(WindowClaimError):
        execute(con, v.plan, model, [Caveat(kind="window", detail="last 1 year")])
