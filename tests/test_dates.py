"""Date order is settled by evidence in the column, never by locale."""

from __future__ import annotations

from nl_insights.ingestion import DateOrder, resolve_date_order


def test_day_first_when_a_first_component_exceeds_12() -> None:
    res = resolve_date_order(["01/02/2011", "13/02/2011", "05/06/2011"])
    assert res.order is DateOrder.DAY_FIRST
    assert "13" in res.evidence  # the disambiguating value is cited


def test_month_first_when_a_second_component_exceeds_12() -> None:
    res = resolve_date_order(["01/02/2011", "01/13/2011"])
    assert res.order is DateOrder.MONTH_FIRST


def test_ambiguous_when_nothing_exceeds_12() -> None:
    res = resolve_date_order(["01/02/2011", "05/06/2011", "07/08/2011"])
    assert res.order is DateOrder.AMBIGUOUS


def test_iso_is_recognised_and_unambiguous() -> None:
    res = resolve_date_order(["2011-02-01", "2011-12-31 08:26:00"])
    assert res.order is DateOrder.ISO


def test_non_dates_are_not_dates() -> None:
    assert resolve_date_order(["WHITE HANGING HEART", "12345", ""]).order is DateOrder.NOT_A_DATE
