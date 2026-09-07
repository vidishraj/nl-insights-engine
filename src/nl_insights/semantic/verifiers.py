"""Deterministic verification — the layer that separates understanding from guessing.

Every LLM claim that implies a *testable* property is tested here in SQL against the
real data. A claim the data contradicts is downgraded or dropped; a claim the data
confirms carries a metric a human can inspect. This is also where the returns
convention and the partition/period structure are *discovered* from evidence rather
than assumed from column names, so the same code works on the enriched file and on
raw UCI.
"""

from __future__ import annotations

import duckdb

from ..ingestion.dates import DateOrder, DateResolution
from ..ingestion.profile import ColumnProfile, Profile
from .model import (
    EntityLabel,
    PartitionDimension,
    PeriodCompleteness,
    ReturnsConvention,
    VerifierResult,
)
from .ontology import EntityKind, Role


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def stored_amount_identity(
    con: duckdb.DuckDBPyConnection, table: str, quantity: str, rate: str, amount: str
) -> VerifierResult:
    """amount ≈ quantity × rate, ROW-WISE. When it holds it binds three roles at once."""
    q, r, a = _q(quantity), _q(rate), _q(amount)
    present = f"{q} IS NOT NULL AND {r} IS NOT NULL AND {a} IS NOT NULL"
    row = con.execute(
        f"SELECT count(*) FILTER (WHERE {present}) AS testable, "
        f"count(*) FILTER (WHERE {present} AND abs({a} - {q} * {r}) <= 0.01) AS matches "
        f"FROM {_q(table)}"
    ).fetchone()
    assert row is not None
    testable, matches = int(row[0]), int(row[1])
    frac = matches / testable if testable else 0.0
    return VerifierResult(
        name="stored_amount_identity",
        passed=testable > 0 and frac >= 0.99,
        detail=f"{amount} == {quantity} * {rate} on {matches}/{testable} rows ({frac:.4f})",
        metrics={"fraction": frac, "testable": float(testable)},
    )


def transaction_key_shares_time(
    con: duckdb.DuckDBPyConnection, table: str, key: str, event_time: str, min_support: int = 5
) -> VerifierResult:
    """Rows sharing the key share an event_time (over multi-row key groups)."""
    k, t = _q(key), _q(event_time)
    row = con.execute(
        f"SELECT count(*) AS support, avg((nt <= 1)::INT)::DOUBLE AS frac "
        f"FROM (SELECT count(*) AS sz, count(DISTINCT {t}) AS nt FROM {_q(table)} GROUP BY {k}) "
        f"WHERE sz > 1"
    ).fetchone()
    assert row is not None
    support = int(row[0])
    frac = float(row[1]) if row[1] is not None else 0.0
    return VerifierResult(
        name="transaction_key_shares_time",
        passed=support >= min_support and frac >= 0.99,
        detail=f"{frac:.4f} of {support} multi-row {key} groups share one {event_time}",
        metrics={"fraction": frac, "support": float(support)},
    )


def additive_quantity_structure(col: ColumnProfile) -> VerifierResult:
    """A quantity must be NUMERIC (a strong disqualifier if not). Integer-dominance is
    recorded as informative evidence but does NOT refute — fractional quantities
    (weights, volumes) are legitimate, so this is a weak signal, not a disproof."""
    if col.numeric is None:
        return VerifierResult(
            name="additive_quantity_structure",
            passed=False,
            detail="not numeric — cannot be a quantity",
        )
    ir = col.numeric.integer_ratio
    return VerifierResult(
        name="additive_quantity_structure",
        passed=True,
        detail=f"integer_ratio={ir:.4f}, negatives={col.numeric.negatives}",
        metrics={"integer_ratio": ir, "negatives": float(col.numeric.negatives)},
    )


def monetary_rate_structure(col: ColumnProfile) -> VerifierResult:
    """A rate must be NUMERIC. Decimal-bearing is recorded as evidence but does NOT
    refute — a whole-number price ($5.00) is a perfectly valid rate, so integer
    dominance is a weak signal, never a disproof."""
    if col.numeric is None:
        return VerifierResult(name="monetary_rate_structure", passed=False, detail="not numeric")
    ir = col.numeric.integer_ratio
    return VerifierResult(
        name="monetary_rate_structure",
        passed=True,
        detail=f"integer_ratio={ir:.4f} (informative; a whole-number rate is still a rate)",
        metrics={"integer_ratio": ir},
    )


def _strptime_formats(order: DateOrder | None) -> list[str]:
    """Candidate strptime formats for a RESOLVED day/month order (with common separators,
    2- or 4-digit year, and optional time). Empty when no order was resolved — we then
    fall back to a bare TRY_CAST rather than guessing the order."""
    if order is DateOrder.MONTH_FIRST:
        dates = ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"]
    elif order is DateOrder.DAY_FIRST:
        dates = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]
    elif order is DateOrder.ISO:
        dates = ["%Y-%m-%d", "%Y/%m/%d"]
    else:
        return []
    times = [" %H:%M:%S", " %H:%M", ""]  # try most-specific first; whole string must match
    return [d + t for d in dates for t in times]


def _fraction_parsed(
    con: duckdb.DuckDBPyConnection, table: str, col: str, expr: str
) -> tuple[int, float]:
    c = _q(col)
    row = con.execute(
        f"SELECT count(*) FILTER (WHERE {c} IS NOT NULL) AS nn, "
        f"count(*) FILTER (WHERE {c} IS NOT NULL AND ({expr}) IS NOT NULL) AS ok "
        f"FROM {_q(table)}"
    ).fetchone()
    assert row is not None
    nn, ok = int(row[0]), int(row[1])
    return nn, (ok / nn if nn else 0.0)


def event_time_temporal(
    con: duckdb.DuckDBPyConnection,
    table: str,
    col: ColumnProfile,
    resolution: DateResolution | None = None,
    min_fraction: float = 0.99,
) -> tuple[VerifierResult, str | None]:
    """An event_time must be a real timestamp the SQL layer can bucket and compare.

    A native date/timestamp passes as-is (no cast). A TEXT column is probed with the
    parser the system ALREADY has — try_strptime built from the date-order the ingester
    detected (``resolution``), falling back to a bare TRY_CAST only when no order was
    resolved. If (nearly) all values parse, it passes and the winning SQL expression is
    returned so the executor emits it. If nothing parses it is a STRONG disproof —
    binding it would let a time query compare raw strings (a silent empty answer) or throw
    — so it is REFUTED and time questions refuse cleanly.

    Returns ``(result, cast_sql)`` where ``cast_sql`` is the parse expression to record
    (None for a native column or a refuted one).
    """
    if col.semantic_type in {"date", "timestamp"}:
        return (
            VerifierResult(
                name="event_time_temporal",
                passed=True,
                detail=f"native {col.semantic_type} type",
                metrics={"fraction": 1.0},
            ),
            None,
        )
    if col.semantic_type not in {"text", "unknown"}:
        return (
            VerifierResult(
                name="event_time_temporal",
                passed=False,
                detail=f"{col.semantic_type} is not a temporal type — cannot bucket by time",
            ),
            None,
        )

    c = _q(col.name)
    order = resolution.order if resolution else None
    formats = _strptime_formats(order)
    # Probe with the detected order first (try_strptime over the candidate formats), then
    # a bare TRY_CAST (handles ISO text). Record whichever clears the threshold.
    best_expr: str | None = None
    best_frac = 0.0
    if formats:
        fmt_list = "[" + ", ".join(_lit(f) for f in formats) + "]"
        expr = f"try_strptime({c}, {fmt_list})"
        nn, frac = _fraction_parsed(con, table, col.name, expr)
        if frac >= min_fraction:
            best_expr, best_frac = expr, frac
    if best_expr is None:
        expr = f"TRY_CAST({c} AS TIMESTAMP)"
        nn, frac = _fraction_parsed(con, table, col.name, expr)
        if frac >= min_fraction:
            best_expr, best_frac = expr, frac
        else:
            best_frac = max(best_frac, frac)

    passed = best_expr is not None
    how = f"order={order.value}" if order else "TRY_CAST"
    detail = (
        f"{best_frac:.4f} of non-null values parse to a timestamp ({how})"
        if passed
        else f"only {best_frac:.4f} parse to a timestamp — not usable for time queries (refused)"
    )
    return (
        VerifierResult(
            name="event_time_temporal",
            passed=passed,
            detail=detail,
            metrics={"fraction": best_frac},
        ),
        best_expr,
    )


def entity_coverage(col: ColumnProfile) -> VerifierResult:
    """Record entity-key coverage as a FACT, not an error (e.g. a quarter of rows null)."""
    coverage = 1.0 - col.null_rate
    return VerifierResult(
        name="entity_coverage",
        passed=True,  # nulls are a coverage fact, never a failure
        detail=f"{coverage:.4f} of rows carry {col.name} ({col.null_rate:.4f} null)",
        metrics={"coverage": coverage, "null_rate": col.null_rate},
    )


def discover_returns(
    con: duckdb.DuckDBPyConnection,
    table: str,
    profile: Profile,
    quantity: str,
    transaction_key: str | None,
    flags: list[str],
    dimensions: list[str],
) -> ReturnsConvention:
    """Discover the return/cancellation convention from correlation with negative
    quantity — never from column names. Prefer an explicit flag or category; fall back
    to a key-prefix convention (raw UCI's ``C``-prefixed cancellations)."""
    qn = _q(quantity)
    tbl = _q(table)
    neg_total = con.execute(f"SELECT count(*) FROM {tbl} WHERE {qn} < 0").fetchone()
    assert neg_total is not None
    if int(neg_total[0]) == 0:
        return ReturnsConvention(kind="none", detail="no negative quantities present")

    # (1) an explicit boolean flag whose true rows are (almost) exactly the negatives.
    #     TRY_CAST, not CAST: a two-valued flag can be TEXT ('charge'/'refund'), and a hard
    #     cast of a non-numeric flag throws mid-ingest. TRY_CAST yields NULL for such a flag,
    #     so it simply does not match here and falls through to the category test below.
    for flag in flags:
        f = _q(flag)
        row = con.execute(
            f"SELECT avg(((TRY_CAST({f} AS INTEGER) = 1) = ({qn} < 0))::INT)::DOUBLE FROM {tbl} "
            f"WHERE {f} IS NOT NULL AND {qn} IS NOT NULL"
        ).fetchone()
        if row and row[0] is not None and float(row[0]) >= 0.99:
            return ReturnsConvention(
                kind="explicit_flag",
                column=flag,
                return_values=["1"],
                detail=f"{flag} aligns with negative quantity ({float(row[0]):.4f})",
                confidence=float(row[0]),
            )

    # (2) an explicit category value whose rows are (almost) all negative. A two-valued TEXT
    #     flag (e.g. line_kind 'charge'/'refund') is a category marker, not a 0/1 flag, so
    #     scan the flag columns here too — a numeric flag already returned above, so only the
    #     text ones reach this loop.
    for dim in dict.fromkeys([*dimensions, *flags]):
        d = _q(dim)
        rows = con.execute(
            f"SELECT {d}::VARCHAR AS v, avg(({qn} < 0)::INT)::DOUBLE AS neg_frac, count(*) AS n "
            f"FROM {tbl} WHERE {d} IS NOT NULL GROUP BY {d} HAVING neg_frac >= 0.9 AND n >= 5"
        ).fetchall()
        if rows:
            values = [str(r[0]) for r in rows]
            return ReturnsConvention(
                kind="explicit_category",
                column=dim,
                return_values=values,
                detail=f"{dim} in {values} are (almost) all negative-quantity rows",
                confidence=float(min(r[1] for r in rows)),
            )

    # (3) derived: negative rows share a leading key character the positive rows lack.
    if transaction_key is not None:
        k = _q(transaction_key)
        row = con.execute(
            f"SELECT left({k}::VARCHAR, 1) AS pfx, avg(({qn} < 0)::INT)::DOUBLE AS neg_frac, "
            f"count(*) AS n FROM {tbl} WHERE {k} IS NOT NULL GROUP BY pfx "
            f"HAVING neg_frac >= 0.9 AND n >= 5 ORDER BY n DESC LIMIT 1"
        ).fetchone()
        if row and row[0] is not None:
            return ReturnsConvention(
                kind="derived_key_prefix",
                column=transaction_key,
                return_values=[str(row[0])],
                detail=f"{transaction_key} values starting {str(row[0])!r} are negative-quantity",
                confidence=float(row[1]),
            )

    return ReturnsConvention(
        kind="derived_negative_quantity",
        column=quantity,
        detail="negatives present but no explicit marker; a negative quantity is the return signal",
        confidence=0.5,
    )


def period_columns(profile: Profile, event_time: str | None, roles: dict[str, Role]) -> set[str]:
    """Dimension columns DERIVED from the event_time (year/quarter/month/…): the
    DIMENSION-role columns the event_time functionally determines. Restricting to
    dimensions keeps sparse/ignored columns from masquerading as period columns."""
    if event_time is None:
        return set()
    return {
        fd.dependent
        for fd in profile.functional_dependencies
        if fd.determinant == event_time
        and fd.strength >= 0.99
        and roles.get(fd.dependent) == Role.DIMENSION
    }


def discover_partition_dimensions(
    con: duckdb.DuckDBPyConnection,
    table: str,
    profile: Profile,
    roles: dict[str, Role],
    periods: set[str],
) -> list[PartitionDimension]:
    """A categorical that partitions the fact table (e.g. a line-type where only one
    value is a real product line). Discovered where a *non-temporal* DIMENSION (nearly)
    determines a FLAG that splits rows into a fact side and a non-fact side."""
    dims = {c for c, r in roles.items() if r == Role.DIMENSION} - periods
    flags = {c for c, r in roles.items() if r == Role.FLAG}
    tbl = _q(table)
    found: dict[str, PartitionDimension] = {}
    for fd in profile.functional_dependencies:
        if fd.determinant not in dims or fd.dependent not in flags or fd.strength < 0.99:
            continue
        d, f = _q(fd.determinant), _q(fd.dependent)
        # TRY_CAST, not CAST: a FLAG can be two-valued TEXT, and a hard cast throws mid-ingest.
        # A non-numeric flag yields NULL here, so it simply forms no partition.
        rows = con.execute(
            f"SELECT {d}::VARCHAR AS v, max(TRY_CAST({f} AS INTEGER)) AS t FROM {tbl} "
            f"WHERE {d} IS NOT NULL GROUP BY {d}"
        ).fetchall()
        true_vals = [str(r[0]) for r in rows if r[1] == 1]
        if not true_vals or len(true_vals) >= len(rows):
            continue  # a genuine partition needs both a fact side and a non-fact side
        candidate = PartitionDimension(
            column=fd.determinant,
            fact_value=true_vals[0] if len(true_vals) == 1 else None,
            via_flag=fd.dependent,
            detail=(
                f"{fd.determinant} partitions the fact table via {fd.dependent}; "
                f"fact rows are {fd.determinant} in {true_vals}"
            ),
        )
        # Prefer a flag that yields a single fact value (the crispest partition).
        existing = found.get(fd.determinant)
        if existing is None or (existing.fact_value is None and candidate.fact_value is not None):
            found[fd.determinant] = candidate
    return list(found.values())


def discover_entity_labels(
    profile: Profile,
    roles: dict[str, Role],
    entities: dict[str, EntityKind | None],
    *,
    min_strength: float = 0.95,
) -> list[EntityLabel]:
    """A DESCRIPTION column functionally tied to an ENTITY_KEY is a LABEL for that
    entity (product description <-> product code). Discovered from an FD in EITHER
    direction — description->key (the description names the code) or key->description
    (the code has a canonical name) — never from the column name. The FD floor is the
    profiler's own recorded floor (>=0.95, support>=5), so any recorded edge qualifies;
    a label lets the binder treat 'group by description' as 'rank that entity'."""
    desc = {c for c, r in roles.items() if r is Role.DESCRIPTION}
    keys = {c for c, r in roles.items() if r is Role.ENTITY_KEY}
    found: dict[tuple[str, str], EntityLabel] = {}
    for fd in profile.functional_dependencies:
        if fd.strength < min_strength:
            continue
        if fd.determinant in desc and fd.dependent in keys:
            label, key = fd.determinant, fd.dependent
        elif fd.determinant in keys and fd.dependent in desc:
            label, key = fd.dependent, fd.determinant
        else:
            continue
        edge = found.get((label, key))
        if edge is None or fd.strength > edge.strength:
            found[(label, key)] = EntityLabel(
                column=label,
                entity_column=key,
                entity=entities.get(key),
                strength=fd.strength,
                support=fd.support,
                detail=(
                    f"{label} labels {key} (FD strength {fd.strength:.3f}, "
                    f"support {fd.support}) — grouping by it ranks the entity"
                ),
            )
    return list(found.values())


def discover_period_completeness(
    profile: Profile,
    roles: dict[str, Role],
    periods: set[str],
) -> list[PeriodCompleteness]:
    """A non-constant boolean flag determined by a *time-derived* period column marks
    period completeness (e.g. a partial first quarter). Period-over-period measures must
    exclude the incomplete periods."""
    flags = {c for c, r in roles.items() if r == Role.FLAG}
    by_name = {c.name: c for c in profile.columns}
    seen: dict[str, PeriodCompleteness] = {}
    for fd in profile.functional_dependencies:
        flag, period = fd.dependent, fd.determinant
        if flag in flags and period in periods and fd.strength >= 0.99:
            col = by_name.get(flag)
            if col and col.distinct_count >= 2 and flag not in seen:  # not constant
                seen[flag] = PeriodCompleteness(
                    flag_column=flag,
                    period_column=period,
                    detail=f"{flag} marks whether a {period} period is complete",
                )
    return list(seen.values())
