"""Header-stripping — the cheapest devastating proof that names are only a weak signal.

Replace a file's header with generic ``col_1..col_N`` and run the same pipeline. If the
system still infers the same structure and answers the same questions, then it was
never leaning on column names — the deterministic layers (profiler, verifiers, binder)
read data, not headers, and the proposer treats a name as one weak hint among the
stats. This turns 'we don't depend on names' from a claim into a CI check.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from ..ingestion.dialect import sniff_dialect


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def strip_headers(source: Path, dest: Path) -> Path:
    """Write a copy of ``source`` whose header row is generic ``col_1..col_N``.

    The values are untouched; only the names are erased. Ingesting ``dest`` therefore
    exercises the pipeline with columns that carry no semantic hint at all.
    """
    dialect = sniff_dialect(source)
    delim = _lit(dialect.delimiter)
    header = "true" if dialect.has_header else "false"
    con = duckdb.connect()
    try:
        read = (
            f"read_csv({_lit(str(source))}, delim={delim}, "
            f"header={header}, all_varchar=true, sample_size=-1)"
        )
        con.execute(f"CREATE TABLE t AS SELECT * FROM {read}")
        original = [r[0] for r in con.execute("DESCRIBE t").fetchall()]
        for i, orig in enumerate(original, start=1):
            con.execute(f"ALTER TABLE t RENAME {_q(orig)} TO {_q(f'col_{i}')}")
        con.execute(f"COPY t TO {_lit(str(dest))} (HEADER, DELIMITER {delim})")
    finally:
        con.close()
    return dest
