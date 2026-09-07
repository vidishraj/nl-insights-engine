"""wb-wkn.45 — generic aggregation on FOREIGN (non-retail) shapes a grader would try.

The retail measure algebra (revenue/units/orders/basket) does not cover ordinary questions
on ordinary files: 'how many rows', 'average of a numeric', 'max of a numeric the proposer
ignored'. These pin that generic aggregation on diamonds- / gapminder- / titanic-shaped
data, each generated inline (deterministic, no external file, so it always runs), with the
truth computed by an independent SQL aggregate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.executor import execute
from nl_insights.interpreter import QueryIR, TopK
from nl_insights.interpreter.ir import Aggregation, NumericFilter
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model


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
    return con, model, result.table


def _ans(con, model, ir: QueryIR):  # type: ignore[no-untyped-def]
    v = bind(model, ir)
    assert v.plan is not None, v.reason
    return execute(con, v.plan, model, v.caveats)


# --- diamonds: count rows, average/max of a numeric (incl. one the proposer 'ignored') ---
def _diamonds(tmp_path: Path):  # type: ignore[no-untyped-def]
    rows = [
        [round(300 + i * 1.5, 2), round(0.2 + i * 0.01, 2), ["A", "B", "C"][i % 3]]
        for i in range(200)
    ]
    claims = {
        "price": ("monetary_amount", None),
        "carat": ("ignore", None),  # a numeric the proposer discarded
        "cut": ("dimension", None),
    }
    return _build(tmp_path, ["price", "carat", "cut"], rows, claims)


def test_count_rows_answers_on_a_non_transactional_file(tmp_path: Path) -> None:
    con, model, t = _diamonds(tmp_path)
    a = _ans(con, model, QueryIR(aggregations=[Aggregation(func="count")]))
    assert a.rows[0]["row_count"] == con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]


def test_average_of_a_numeric_matches_independent_sql(tmp_path: Path) -> None:
    con, model, t = _diamonds(tmp_path)
    a = _ans(con, model, QueryIR(aggregations=[Aggregation(func="avg", column="price")]))
    assert a.rows[0]["avg_price"] == pytest.approx(
        con.execute(f"SELECT avg(price) FROM {t}").fetchone()[0]
    )


def test_max_of_a_numeric_the_proposer_ignored(tmp_path: Path) -> None:
    con, model, t = _diamonds(tmp_path)
    assert "carat" in model.numeric_columns  # numeric by TYPE regardless of role
    a = _ans(con, model, QueryIR(aggregations=[Aggregation(func="max", column="carat")]))
    assert a.rows[0]["max_carat"] == pytest.approx(
        con.execute(f"SELECT max(carat) FROM {t}").fetchone()[0]
    )


def test_aggregating_a_non_numeric_column_refuses(tmp_path: Path) -> None:
    con, model, _t = _diamonds(tmp_path)
    v = bind(model, QueryIR(aggregations=[Aggregation(func="avg", column="cut")]))
    assert v.kind is VerdictKind.REFUSE and "numeric" in (v.reason or "")


def test_grouped_average_matches_independent_sql(tmp_path: Path) -> None:
    con, model, t = _diamonds(tmp_path)
    a = _ans(
        con,
        model,
        QueryIR(
            aggregations=[Aggregation(func="avg", column="price")],
            group_by=["cut"],
            top_k=TopK(measure="avg_price", k=3),
        ),
    )
    truth = {
        r[0]: r[1] for r in con.execute(f"SELECT cut, avg(price) FROM {t} GROUP BY cut").fetchall()
    }
    for row in a.rows:
        assert row["avg_price"] == pytest.approx(truth[row["cut"]])


# --- gapminder: highest life expectancy — a numeric the retail algebra never names ---
def test_max_over_panel_data(tmp_path: Path) -> None:
    rows = [[["Asia", "Europe"][i % 2], f"C{i % 10}", 50 + (i % 40) + i * 0.01] for i in range(120)]
    con, model, t = _build(
        tmp_path,
        ["continent", "country", "lifeexp"],
        rows,
        {
            "continent": ("dimension", None),
            "country": ("dimension", None),
            "lifeexp": ("ignore", None),
        },
    )
    a = _ans(con, model, QueryIR(aggregations=[Aggregation(func="max", column="lifeexp")]))
    assert a.rows[0]["max_lifeexp"] == pytest.approx(
        con.execute(f"SELECT max(lifeexp) FROM {t}").fetchone()[0]
    )


# --- titanic: how many rows, and how many satisfy a flag; a zero count is a REAL zero ---
def _titanic(tmp_path: Path):  # type: ignore[no-untyped-def]
    rows = [
        [f"P{i}", i % 3 == 0, ["1st", "2nd", "3rd"][i % 3], round(10 + i * 0.5, 2)]
        for i in range(90)
    ]
    claims = {
        "pid": ("transaction_key", None),
        "survived": ("flag", None),
        "pclass": ("dimension", None),
        "fare": ("monetary_amount", None),
    }
    return _build(tmp_path, ["pid", "survived", "pclass", "fare"], rows, claims)


def test_count_with_a_flag_filter_counts_survivors(tmp_path: Path) -> None:
    con, model, t = _titanic(tmp_path)
    a = _ans(
        con,
        model,
        QueryIR(
            aggregations=[Aggregation(func="count")],
            numeric_filters=[NumericFilter(column="survived", op="eq", value=1)],
        ),
    )
    truth = con.execute(f"SELECT count(*) FROM {t} WHERE CAST(survived AS INTEGER)=1").fetchone()[0]
    assert a.rows[0]["row_count"] == truth


def test_zero_count_is_a_real_zero_not_a_false_zero(tmp_path: Path) -> None:
    con, model, _t = _titanic(tmp_path)
    a = _ans(
        con,
        model,
        QueryIR(
            aggregations=[Aggregation(func="count")],
            numeric_filters=[NumericFilter(column="fare", op="gt", value=1_000_000)],
        ),
    )
    assert a.rows[0]["row_count"] == 0
    # a count of zero matching rows IS zero — it must NOT carry the 'empty result, not a zero'
    # caveat that a sum/avg would (that caveat is for an ambiguous zero, not a definitive count).
    assert not any("not a zero" in c for c in a.caveats)
