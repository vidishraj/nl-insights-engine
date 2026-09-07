"""Ingest a CSV into a per-dataset DuckDB file and profile it.

One DuckDB file *per dataset* is a deliberate concurrency choice: ingesting or
querying dataset A never contends with dataset B, and it is the honest answer to the
brief's "how does this behave under concurrent load" — isolation by construction
rather than a lock we have to reason about.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .dates import DateOrder, DateResolution, resolve_date_order
from .dialect import Dialect, sniff_dialect
from .profile import Profile, compute_profile

# Sample broadly (reservoir), not the head: a date-sorted file hides the
# disambiguating value (a day > 12) far from the top.
_DATE_SAMPLE_ROWS = 20_000

# A narration/cancellation hook: called at each internal step with (stage, message).
# It may RAISE to abort (the job layer uses this for stage-boundary cancellation), and
# it is what lets a UI narrate the dataset assembling. Defaults to a no-op.
StageFn = Callable[[str, str], None]


def _noop(_stage: str, _message: str) -> None:
    return None


def dataset_id_for(source: Path, name: str | None = None) -> str:
    """The dataset id an ingest of ``source`` (optionally named) will use.

    Exposed so the job layer can derive the id BEFORE running — to serialise two
    ingests of the same dataset and to key idempotency — without re-implementing it.
    """
    return _dataset_id(source, name)


class IngestionError(RuntimeError):
    """Raised when a CSV cannot be loaded into DuckDB."""


@dataclass(frozen=True)
class IngestResult:
    dataset_id: str
    duckdb_path: Path
    table: str
    dialect: Dialect
    profile: Profile
    # Evidence-based date order for any text column that looks like a date.
    date_orders: dict[str, DateResolution]


def _dataset_id(source: Path, name: str | None) -> str:
    if name:
        return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", source.stem).strip("_") or "dataset"
    digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:8]
    return f"{stem}_{digest}"


def _sql_literal(ch: str) -> str:
    """A single-quoted SQL literal containing ``ch`` (used for the CSV options)."""
    return "'" + ch.replace("'", "''") + "'"


# DuckDB read_csv accepts utf-8, utf-16, latin-1, and cp1252; map the Python-only codec name
# the sniffer may return (utf-8-sig, from a BOM) to one DuckDB understands.
_DUCKDB_ENCODING = {"utf-8-sig": "utf-8"}


def _duckdb_encoding(sniffed: str) -> str:
    return _DUCKDB_ENCODING.get(sniffed, sniffed)


def ingest(
    source: Path,
    *,
    data_dir: Path,
    name: str | None = None,
    table: str = "dataset",
    on_stage: StageFn | None = None,
) -> IngestResult:
    # on_stage narrates each internal step (and may raise to abort — the job layer uses
    # it for stage-boundary cancellation). Called BEFORE each step, so a caller sees
    # "sniffing" while sniffing happens, not after.
    emit = on_stage or _noop
    if not source.exists():
        raise IngestionError(f"No such file: {source.name}")

    emit("sniffing", "detecting delimiter, quote and header")
    dialect = sniff_dialect(source)
    dataset_id = _dataset_id(source, name)
    data_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = data_dir / f"{dataset_id}.duckdb"
    if duckdb_path.exists():  # rebuild fresh; per-dataset file is owned by this ingest
        duckdb_path.unlink()

    con = duckdb.connect(str(duckdb_path))
    try:
        # Pass the SNIFFED encoding through to DuckDB — never let it re-guess and die on a
        # non-UTF-8 file (a cp1252/Latin-1 retail export is the documented normal shape). If the
        # sniffed encoding does not take, fall back to latin-1, which decodes ANY byte, rather
        # than fail to load a file DuckDB can read: 'it would not load my file' is worse than
        # any wrong number. A genuinely ragged file still fails under every encoding, so the
        # fallback only rescues an encoding miss, never masks a structural problem.
        encodings = [_duckdb_encoding(dialect.encoding)]
        if "latin-1" not in encodings:
            encodings.append("latin-1")

        emit("loading", "loading rows into DuckDB with full-file type inference")
        last_exc: duckdb.Error | None = None
        used_encoding = encodings[0]
        for enc in encodings:
            # sample_size=-1 → type inference scans the FULL file, not just a head sample, so a
            # type that only breaks on row 400k is caught at load, not at query time.
            read = (
                f"read_csv(?, "
                f"delim={_sql_literal(dialect.delimiter)}, "
                f"quote={_sql_literal(dialect.quotechar)}, "
                f"header={'true' if dialect.has_header else 'false'}, "
                f"encoding={_sql_literal(enc)}, "
                f"sample_size=-1)"
            )
            try:
                con.execute(f"DROP TABLE IF EXISTS {table}")
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM {read}", [str(source)])
                last_exc = None
                used_encoding = enc
                break
            except duckdb.Error as exc:
                last_exc = exc
        if last_exc is not None:
            # Keep the parser's diagnosis (e.g. 'ragged near line 2') but never the
            # server-side absolute path — only the uploaded file's basename.
            raise IngestionError(
                f"could not load {source.name} as a CSV: {last_exc}"
            ) from last_exc

        emit("profiling", "profiling types, cardinalities and functional dependencies")
        profile = compute_profile(con, table, dataset_id=dataset_id, source=str(source))
        if profile.row_count == 0:
            # A header-only file loads "successfully" with zero rows; refuse by name
            # rather than let downstream produce confident answers over nothing.
            raise IngestionError(f"{source.name} has a header but no data rows — nothing to query.")
        # Resolve date order from the ORIGINAL file strings (a DuckDB DATE column has
        # been normalised to ISO, erasing the day/month order we need), sampled
        # broadly so a date-sorted file still yields its disambiguating value.
        emit("resolving_dates", "resolving day/month order from a broad sample")
        date_orders = _resolve_dates(
            con, source, dialect, [c.name for c in profile.columns], encoding=used_encoding
        )
    finally:
        con.close()

    return IngestResult(
        dataset_id=dataset_id,
        duckdb_path=duckdb_path,
        table=table,
        dialect=dialect,
        profile=profile,
        date_orders=date_orders,
    )


def _resolve_dates(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    dialect: Dialect,
    column_names: list[str],
    *,
    encoding: str,
) -> dict[str, DateResolution]:
    """Resolve date order per column from a broad reservoir sample of raw strings. Reads with
    the SAME encoding the main load succeeded under, so a cp1252 file is not re-read as UTF-8
    and made to fail after the load already worked."""
    src_lit = str(source).replace("'", "''")
    read = (
        f"read_csv('{src_lit}', "
        f"delim={_sql_literal(dialect.delimiter)}, "
        f"quote={_sql_literal(dialect.quotechar)}, "
        f"header={'true' if dialect.has_header else 'false'}, "
        f"encoding={_sql_literal(encoding)}, "
        f"all_varchar=true, sample_size=-1)"
    )
    sample = con.execute(f"SELECT * FROM {read} USING SAMPLE {_DATE_SAMPLE_ROWS} ROWS").fetchall()

    resolutions: dict[str, DateResolution] = {}
    for idx, name in enumerate(column_names):
        values = [row[idx] for row in sample if idx < len(row) and row[idx] is not None]
        resolution = resolve_date_order(values)
        if resolution.order != DateOrder.NOT_A_DATE:
            resolutions[name] = resolution
    return resolutions
