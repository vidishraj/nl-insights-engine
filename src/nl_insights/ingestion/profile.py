"""Deterministic profiler — the evidence base the semantic layer later reasons over.

Every number here is *computed by SQL over the whole table*, never sampled and never
guessed by a model. That is what makes the layer above it checkable: a claim like
"this column is the product identifier" can be tested against real cardinality,
uniqueness, and functional dependencies rather than a column name.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel

# DuckDB type name -> coarse semantic type. Kept dataset-agnostic: it reads the
# storage type, not the column name.
_INTEGER_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
}
_DECIMAL_PREFIXES = ("DECIMAL", "DOUBLE", "FLOAT", "REAL")


class ValueCount(BaseModel):
    value: str | None
    count: int


class NumericStats(BaseModel):
    negatives: int
    zeros: int
    positives: int
    integer_ratio: float  # fraction of non-null values that are whole numbers


class ColumnProfile(BaseModel):
    name: str
    storage_type: str
    semantic_type: str  # integer | decimal | boolean | date | timestamp | text | unknown
    null_count: int
    null_rate: float
    distinct_count: int
    uniqueness_ratio: float  # distinct / non-null
    min_value: str | None
    max_value: str | None
    numeric: NumericStats | None = None
    top_values: list[ValueCount]
    samples: list[str]


class FunctionalDependency(BaseModel):
    """``determinant`` (mostly) determines ``dependent``.

    ``strength`` is the fraction of determinant groups that map to a single dependent
    value, computed **only over groups with more than one row**. A one-row group
    trivially has one distinct value of *every* column, so counting singletons would
    inflate strength for a near-unique column into a spurious ~0.94 against anything —
    it must not count. ``support`` is how many multi-row groups the strength was
    measured over, so a consumer can weigh 0.99-over-25000 differently from 0.99-over-3.
    """

    determinant: str
    dependent: str
    strength: float
    support: int


class Profile(BaseModel):
    dataset_id: str
    source: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    functional_dependencies: list[FunctionalDependency]


def _q(identifier: str) -> str:
    """Quote a SQL identifier safely."""
    return '"' + identifier.replace('"', '""') + '"'


def _semantic_type(storage_type: str) -> str:
    upper = storage_type.upper()
    if upper in _INTEGER_TYPES:
        return "integer"
    if upper.startswith(_DECIMAL_PREFIXES):
        return "decimal"
    if upper == "BOOLEAN":
        return "boolean"
    if upper == "DATE":
        return "date"
    if upper.startswith("TIMESTAMP"):
        return "timestamp"
    if upper in {"VARCHAR", "TEXT", "STRING"}:
        return "text"
    return "unknown"


def compute_profile(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    dataset_id: str,
    source: str,
    top_n: int = 10,
    sample_n: int = 5,
    fd_min_strength: float = 0.95,
    fd_min_support: int = 5,
    fd_max_groups: int = 100_000,
    fd_max_determinants: int = 25,
) -> Profile:
    tq = _q(table)
    row_count: int = con.execute(f"SELECT count(*) FROM {tq}").fetchone()[0]  # type: ignore[index]

    schema = con.execute(f"DESCRIBE {tq}").fetchall()
    names = [r[0] for r in schema]
    types = {r[0]: r[1] for r in schema}

    columns = _basic_stats(con, tq, names, types, row_count)
    _attach_numeric_stats(con, tq, columns)
    for col in columns:
        col.top_values = _top_values(con, tq, col.name, top_n)
        col.samples = _samples(con, tq, col.name, sample_n)

    fds = _functional_dependencies(
        con, tq, columns, fd_min_strength, fd_min_support, fd_max_groups, fd_max_determinants
    )

    return Profile(
        dataset_id=dataset_id,
        source=source,
        row_count=row_count,
        column_count=len(columns),
        columns=columns,
        functional_dependencies=fds,
    )


def _basic_stats(
    con: duckdb.DuckDBPyConnection,
    tq: str,
    names: list[str],
    types: dict[str, str],
    row_count: int,
) -> list[ColumnProfile]:
    # One scan computes non-null count, distinct count, min and max for every column.
    selects: list[str] = []
    for i, name in enumerate(names):
        c = _q(name)
        selects += [
            f"count({c}) AS nn_{i}",
            f"count(DISTINCT {c}) AS nd_{i}",
            f"min({c})::VARCHAR AS mn_{i}",
            f"max({c})::VARCHAR AS mx_{i}",
        ]
    row = con.execute(f"SELECT {', '.join(selects)} FROM {tq}").fetchone()
    assert row is not None

    columns: list[ColumnProfile] = []
    for i, name in enumerate(names):
        non_null = int(row[4 * i])
        distinct = int(row[4 * i + 1])
        min_v = row[4 * i + 2]
        max_v = row[4 * i + 3]
        null_count = row_count - non_null
        columns.append(
            ColumnProfile(
                name=name,
                storage_type=types[name],
                semantic_type=_semantic_type(types[name]),
                null_count=null_count,
                null_rate=(null_count / row_count) if row_count else 0.0,
                distinct_count=distinct,
                uniqueness_ratio=(distinct / non_null) if non_null else 0.0,
                min_value=min_v,
                max_value=max_v,
                top_values=[],
                samples=[],
            )
        )
    return columns


def _attach_numeric_stats(
    con: duckdb.DuckDBPyConnection,
    tq: str,
    columns: list[ColumnProfile],
) -> None:
    numeric = [c for c in columns if c.semantic_type in {"integer", "decimal"}]
    if not numeric:
        return
    selects: list[str] = []
    for i, col in enumerate(numeric):
        c = _q(col.name)
        selects += [
            f"count(*) FILTER (WHERE {c} < 0) AS neg_{i}",
            f"count(*) FILTER (WHERE {c} = 0) AS zero_{i}",
            f"count(*) FILTER (WHERE {c} > 0) AS pos_{i}",
            f"count(*) FILTER (WHERE {c} IS NOT NULL AND {c} = floor({c})) AS whole_{i}",
            f"count({c}) AS nn_{i}",
        ]
    row = con.execute(f"SELECT {', '.join(selects)} FROM {tq}").fetchone()
    assert row is not None
    for i, col in enumerate(numeric):
        neg, zero, pos, whole, nn = (int(row[5 * i + k]) for k in range(5))
        col.numeric = NumericStats(
            negatives=neg,
            zeros=zero,
            positives=pos,
            integer_ratio=(whole / nn) if nn else 0.0,
        )


def _top_values(con: duckdb.DuckDBPyConnection, tq: str, name: str, top_n: int) -> list[ValueCount]:
    c = _q(name)
    rows = con.execute(
        f"SELECT {c}::VARCHAR AS v, count(*) AS n FROM {tq} WHERE {c} IS NOT NULL "
        f"GROUP BY {c} ORDER BY n DESC, v LIMIT {top_n}"
    ).fetchall()
    return [ValueCount(value=r[0], count=int(r[1])) for r in rows]


def _samples(con: duckdb.DuckDBPyConnection, tq: str, name: str, sample_n: int) -> list[str]:
    # ORDER BY makes the sample DETERMINISTIC — DISTINCT+LIMIT without it returns an
    # arbitrary order, which would make the evidence prompt (and any recorded LLM cassette
    # keyed on it) non-reproducible across ingests.
    c = _q(name)
    rows = con.execute(
        f"SELECT DISTINCT {c}::VARCHAR AS v FROM {tq} WHERE {c} IS NOT NULL "
        f"ORDER BY v LIMIT {sample_n}"
    ).fetchall()
    return [r[0] for r in rows]


def _functional_dependencies(
    con: duckdb.DuckDBPyConnection,
    tq: str,
    columns: list[ColumnProfile],
    min_strength: float,
    min_support: int,
    max_groups: int,
    max_determinants: int,
) -> list[FunctionalDependency]:
    # Any *repeated* column can be a determinant — including a high-cardinality
    # transaction key (tens of thousands distinct, but each value recurs, so its
    # uniqueness ratio is low). We do NOT cap on cardinality: that would exclude keys
    # by construction. We exclude only near-unique row ids (they determine everything
    # trivially) and columns so wide that grouping them is too costly.
    candidates = [
        c for c in columns if 2 <= c.distinct_count <= max_groups and c.uniqueness_ratio < 0.99
    ]
    # Most-repeated first: keys and categoricals both have low uniqueness, so this
    # keeps them ahead of anything borderline when the budget bites.
    candidates.sort(key=lambda c: c.uniqueness_ratio)
    candidates = candidates[:max_determinants]

    fds: list[FunctionalDependency] = []
    names = [c.name for c in columns]
    for det in candidates:
        others = [n for n in names if n != det.name]
        if not others:
            continue
        # One scan per determinant: group by it, then over the MULTI-ROW groups only
        # (sz > 1) measure, per dependent, the fraction that map to a single value.
        # Singleton groups are unfalsifiable — a one-row group trivially has one
        # distinct value of everything — so they are excluded; `support` records how
        # many multi-row groups the strength was computed over.
        inner = "count(*) AS sz, " + ", ".join(
            f"count(DISTINCT {_q(o)}) AS d{j}" for j, o in enumerate(others)
        )
        strengths = ", ".join(f"avg((d{j} <= 1)::INT)::DOUBLE AS s{j}" for j in range(len(others)))
        row = con.execute(
            f"SELECT count(*) AS support, {strengths} "
            f"FROM (SELECT {inner} FROM {tq} GROUP BY {_q(det.name)}) WHERE sz > 1"
        ).fetchone()
        assert row is not None
        support = int(row[0])
        if support < min_support:
            continue  # too few multi-row groups to claim anything
        for j, dep in enumerate(others):
            strength = float(row[j + 1]) if row[j + 1] is not None else 0.0
            if strength >= min_strength:
                fds.append(
                    FunctionalDependency(
                        determinant=det.name, dependent=dep, strength=strength, support=support
                    )
                )
    fds.sort(key=lambda f: (f.strength, f.support), reverse=True)
    return fds
