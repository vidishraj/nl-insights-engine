"""Compile a BoundPlan into DuckDB SQL — deterministic, every clause traceable to a
plan node. The binder has already validated references and injected the correctness
defaults (partition filter, incomplete-period exclusion), so compilation is mechanical.
"""

from __future__ import annotations

from ..binder.verdict import BoundPlan
from ..interpreter.ir import CategoricalFilter, NumericFilter, TimeWindow

_NUM_OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
_GRAIN_FMT = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m", "year": "%Y"}
_GRAIN_UNIT = {"day": "days", "week": "weeks", "month": "months", "year": "years"}


def last_n_interval(win: TimeWindow) -> str | None:
    """The DuckDB INTERVAL literal for a `last N <grain>` window, or None if the window
    does not carry a rolling `last_n`. A quarter is three months."""
    if not (win.last_n and win.grain):
        return None
    if win.grain == "quarter":
        return f"INTERVAL '{win.last_n * 3} months'"
    return f"INTERVAL '{win.last_n} {_GRAIN_UNIT.get(win.grain, 'years')}'"


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def complete_period_clause(flag: str) -> str:
    """The predicate that keeps only COMPLETE periods, in ONE place so the two compile sites
    and the executor's structural guard match the exact clause (not just the column name).
    TRY_CAST, not CAST: a completeness flag can be TEXT ('complete'/'true'), and a hard cast
    would throw mid-query on it (the same family as the returns TRY_CAST fix); a text flag is
    also matched against generic truthy tokens so it is not silently excluded."""
    q = _q(flag)
    return (
        f"(TRY_CAST({q} AS INTEGER) = 1 OR "
        f"lower(TRIM({q}::VARCHAR)) IN ('true', 'complete', 'yes', 't', 'y'))"
    )


_CAT_OPS = {"in": "IN", "not_in": "NOT IN"}


def _categorical(f: CategoricalFilter) -> str:
    vals = ", ".join(_lit(v) for v in f.values) or "NULL"
    # Direct index, not `.get(op, "IN")`: an unmapped op must raise here, never silently
    # select the OPPOSITE set. The IR Literal and the binder both keep this unreachable in
    # practice; this is the third, structural line of defence.
    op = _CAT_OPS[f.op]
    return f"{_q(f.column)}::VARCHAR {op} ({vals})"


def _numeric(f: NumericFilter) -> str:
    if f.op == "between" and f.value2 is not None:
        return f"{_q(f.column)} BETWEEN {f.value} AND {f.value2}"
    return f"{_q(f.column)} {_NUM_OPS[f.op]} {f.value}"


def _period_expr(event_time: str, grain: str) -> str:
    """A label for the period a row falls in, at the given grain. ``event_time`` is a
    ready SQL expression (a quoted column, or a CAST of a text date), not a bare name."""
    et = event_time
    if grain == "quarter":
        return f"(strftime({et}, '%Y') || '-Q' || CAST(quarter({et}) AS VARCHAR))"
    fmt = _GRAIN_FMT.get(grain, "%Y-%m")
    return f"strftime({et}, '{fmt}')"


def _time(win: TimeWindow, event_time: str, table: str) -> list[str]:
    et = event_time  # already a SQL expression (quoted column or CAST)
    clauses: list[str] = []
    if win.named_period and win.grain:
        clauses.append(f"{_period_expr(event_time, win.grain)} = {_lit(win.named_period)}")
    elif win.named_period:
        clauses.append(f"strftime({et}, '%Y-%m') = {_lit(win.named_period)}")
    if win.start:
        clauses.append(f"{et} >= {_lit(win.start)}")
    if win.end:
        clauses.append(f"{et} < {_lit(win.end)}")
    # A rolling `last N <grain>` window, anchored on the DATA's latest event_time (not
    # today — today would return zero rows for a historical file). The executor discloses the
    # anchor and whether the window covers the whole span; here we just emit the real clause,
    # so a claimed window is never absent from the SQL.
    interval = last_n_interval(win)
    if interval is not None:
        clauses.append(f"{et} >= (SELECT max({et}) FROM {_q(table)}) - {interval}")
    return clauses


def compile_sql(plan: BoundPlan, event_time: str | None) -> str:
    ir = plan.ir
    group_terms = [_q(g) for g in plan.group_by]
    select: list[str] = list(group_terms)

    # A period-over-period comparison is NOT compiled here — it has its own executor branch
    # (_compile_period_comparison) that pivots two periods and computes the ranked growth.
    # The old code here only added a period label and left the delta "to the reader".

    for m in plan.measures:
        select.append(f"{m.sql} AS {_q(m.name)}")
    if ir.distinct_count_of:
        select.append(f"count(DISTINCT {_q(ir.distinct_count_of)}) AS {_q('distinct_count')}")
    if not select:
        select = ["count(*) AS row_count"]

    where: list[str] = []
    for cf in [*ir.categorical_filters, *plan.applied_filters]:
        where.append(_categorical(cf))
    for nf in ir.numeric_filters:
        where.append(_numeric(nf))
    if ir.time and event_time:
        where.extend(_time(ir.time, event_time, plan.table))
    if plan.require_complete_period_flag:
        where.append(complete_period_clause(plan.require_complete_period_flag))

    sql = f"SELECT {', '.join(select)} FROM {_q(plan.table)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group_terms:
        sql += " GROUP BY " + ", ".join(group_terms)
    if ir.top_k:
        direction = "ASC" if ir.top_k.direction == "asc" else "DESC"
        sql += f" ORDER BY {_q(ir.top_k.measure)} {direction} LIMIT {ir.top_k.k}"
    return sql
