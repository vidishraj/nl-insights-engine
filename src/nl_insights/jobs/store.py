"""The dataset registry: the READY, queryable understanding of each dataset.

A dataset becomes visible here only after its DuckDB file has been built to a staging
path and atomically promoted, and its semantic model built — so a query never sees a
half-loaded dataset. Re-ingesting a dataset replaces its entry in one assignment
(atomic on the event loop thread); an in-flight query either used the previous file
(its open fd survives the os.replace on POSIX) or opens the new one. There is never a
window where the canonical path holds a partial file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
from pydantic import BaseModel

from ..semantic.model import CategoricalValues, SemanticModel

_log = logging.getLogger(__name__)

# The sidecar is the "understanding of record": DuckDB holds the data, this JSON holds the
# verified semantic model, written beside the data file so a restart can rehydrate it.
_SIDECAR_SUFFIX = ".dataset.json"
# The semantic-model schema version this build understands. A sidecar written by a DIFFERENT
# version is refused on load rather than trusted — the field exists for exactly this.
CURRENT_MODEL_VERSION: int = SemanticModel.model_fields["version"].default
# Older versions we can migrate FORWARD by re-deriving the missing fields from the DuckDB file
# (rather than refusing and forcing a re-upload). v1 -> v2 re-derives `categorical_values` (a
# SELECT DISTINCT); v2 -> v3 additionally re-derives `numeric_columns` (the DuckDB column
# types) — both no-LLM, no-loss, so an older dataset upgrades in place without a re-upload.
_MIGRATABLE_VERSIONS = {1, 2}
_TOP_N = 10  # matches the profiler's top-values sample size, so exhaustiveness agrees


class Dataset(BaseModel):
    """A ready dataset: where its DuckDB file is and what we understand about it."""

    dataset_id: str
    table: str
    duckdb_path: str
    row_count: int
    model: SemanticModel

    def connect(self) -> duckdb.DuckDBPyConnection:
        """A READ-ONLY connection — queries never mutate a dataset, and read_only lets
        concurrent queries share the file without contending."""
        return duckdb.connect(self.duckdb_path, read_only=True)


class DatasetStore:
    """In-memory registry of ready datasets. Reads and the single-assignment writes are
    atomic on the event-loop thread, so no lock is needed for correctness here."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}

    def put(self, dataset: Dataset) -> None:
        self._datasets[dataset.dataset_id] = dataset  # atomic replace

    def get(self, dataset_id: str) -> Dataset | None:
        return self._datasets.get(dataset_id)

    def ids(self) -> list[str]:
        return sorted(self._datasets)

    def summaries(self) -> list[dict[str, object]]:
        return [
            {
                "dataset_id": d.dataset_id,
                "row_count": d.row_count,
                "columns": len(d.model.bindings),
                "measures": [m.name for m in d.model.measures if m.available],
            }
            for d in (self._datasets[i] for i in self.ids())
        ]


def promote(staging_path: Path, canonical_path: Path) -> None:
    """Atomically move a fully-built staging DuckDB file into its canonical location.

    ``os.replace`` (via ``Path.replace``) is atomic within a filesystem, so a reader
    either sees the old file or the new one — never a partial. The job layer stages
    inside the data dir to guarantee the same filesystem.
    """
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path.replace(canonical_path)


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sidecar_for(duckdb_path: str | Path) -> Path:
    p = Path(duckdb_path)
    return p.with_name(p.stem + _SIDECAR_SUFFIX)


def _identity_mismatch(dataset: Dataset) -> str | None:
    """Return a reason the persisted model does NOT describe the data file it points at, or
    None if it does. Existence is not identity: if the table were rebuilt underneath (a
    different schema, or a different size), the model would describe columns that no longer
    exist and 'the model is verified against the data' would be silently false after a
    restart. Check the row_count AND the column set — a size match can coincide while the
    schema drifts, and the model fundamentally describes COLUMNS."""
    path = Path(dataset.duckdb_path)
    if not path.exists():
        return "its DuckDB file is gone"
    tbl = _q(dataset.table)
    try:
        con = duckdb.connect(str(path), read_only=True)
        try:
            row = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()
            actual_rows = int(row[0]) if row else -1
            actual_cols = {d[0] for d in con.execute(f"SELECT * FROM {tbl} LIMIT 0").description}
        finally:
            con.close()
    except (duckdb.Error, OSError):  # unreadable file, or the described table is gone
        return "its DuckDB file could not be read as the described table"
    if actual_rows != dataset.row_count:
        return f"row_count {dataset.row_count} does not match the data ({actual_rows})"
    missing = sorted({b.column for b in dataset.model.bindings} - actual_cols)
    if missing:
        return f"the model describes columns absent from the data: {', '.join(missing)}"
    return None


def write_sidecar(dataset: Dataset) -> None:
    """Persist a ready dataset's understanding beside its DuckDB file, so a restart can
    rehydrate it instead of dropping every visitor dataset. Written atomically (tmp +
    replace) so a crash mid-write never leaves a partial sidecar a reader would trust."""
    sidecar = _sidecar_for(dataset.duckdb_path)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(dataset.model_dump_json(), encoding="utf-8")
    tmp.replace(sidecar)


def _rederive_categorical_values(
    con: duckdb.DuckDBPyConnection, table: str, columns: list[str]
) -> dict[str, CategoricalValues]:
    """Recompute each bound column's observed values from the data — the same SELECT the
    profiler runs, so exhaustiveness is decided identically (distinct_count <= sample size)."""
    out: dict[str, CategoricalValues] = {}
    tbl = _q(table)
    for col in columns:
        c = _q(col)
        try:
            rows = con.execute(
                f"SELECT {c}::VARCHAR AS v, count(*) AS n FROM {tbl} WHERE {c} IS NOT NULL "
                f"GROUP BY v ORDER BY n DESC, v LIMIT {_TOP_N}"
            ).fetchall()
            distinct = con.execute(f"SELECT count(DISTINCT {c}) FROM {tbl}").fetchone()
        except (duckdb.Error, OSError):
            continue
        values = [str(r[0]) for r in rows]
        if values and distinct is not None:
            out[col] = CategoricalValues(values=values, exhaustive=int(distinct[0]) <= len(values))
    return out


def _rederive_numeric_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Recompute the numeric columns from the DuckDB column TYPES — a v3 field the binder uses
    to validate a generic aggregation. No LLM: the stored type is ground truth."""
    try:
        rows = con.execute(f"DESCRIBE {_q(table)}").fetchall()
    except (duckdb.Error, OSError):
        return []
    numeric = (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
        "REAL",
    )
    out: list[str] = []
    for r in rows:
        name, col_type = str(r[0]), str(r[1]).upper()
        if col_type.startswith("DECIMAL") or col_type in numeric:
            out.append(name)
    return out


def _migrate_in_place(dataset: Dataset) -> None:
    """Upgrade an older sidecar to the current schema by RE-DERIVING the fields the current
    binder relies on from the still-present DuckDB file, then rewrite the sidecar so the
    migration is permanent. Best-effort on the rewrite — the in-memory model is already
    upgraded, so the dataset works this session even if the disk upgrade fails."""
    try:
        con = duckdb.connect(dataset.duckdb_path, read_only=True)
        try:
            cols = [b.column for b in dataset.model.bindings if not b.refuted]
            dataset.model.categorical_values = _rederive_categorical_values(
                con, dataset.table, cols
            )
            dataset.model.numeric_columns = _rederive_numeric_columns(con, dataset.table)
        finally:
            con.close()
    except (duckdb.Error, OSError) as exc:
        # RECOVERABLE, not catastrophic: the dataset is still put into the store (below, by the
        # caller) with the fields it loaded with; only the NEW capability that depends on the
        # re-derived field degrades (a generic aggregation refuses; the retail path is
        # untouched), the version is NOT bumped, so the migration is retried on the next
        # restart. Read-only + identity-checked first, so nothing is corrupted.
        _log.warning("could not re-derive values for %s: %s", dataset.dataset_id, exc)
        return
    from_version = dataset.model.version
    dataset.model.version = CURRENT_MODEL_VERSION
    # OBSERVABILITY: log that the migration RAN and WHAT it produced, so a deploy can confirm
    # the migrate-in-place path actually executed against the real data — not merely that the
    # service came up. A process that started is not evidence the migration worked.
    _log.info(
        "migrated dataset %s from model v%s to v%d: re-derived %d numeric columns, "
        "%d categorical value sets",
        dataset.dataset_id,
        from_version,
        CURRENT_MODEL_VERSION,
        len(dataset.model.numeric_columns),
        len(dataset.model.categorical_values),
    )
    try:
        write_sidecar(dataset)
    except OSError:
        _log.warning(
            "re-derived %s in memory but could not upgrade its sidecar", dataset.dataset_id
        )


def load_persisted(store: DatasetStore, data_dir: Path) -> int:
    """Rehydrate ready datasets from their sidecars on startup. A sidecar whose semantic
    model version this build does not recognise is REFUSED with a clear log (not trusted);
    one whose DuckDB file is gone, or that will not parse, is skipped. Returns the count
    loaded. This is what makes the model a persisted ARTIFACT, not a dict that dies with
    the process."""
    if not data_dir.exists():
        return 0
    loaded = 0
    for sidecar in sorted(data_dir.glob(f"*{_SIDECAR_SUFFIX}")):
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _log.warning("skipping unreadable dataset sidecar %s", sidecar.name)
            continue
        version = (raw.get("model") or {}).get("version")
        # A sidecar written by an OLDER-but-known version loads fine structurally but is missing
        # fields the binder now depends on — trusting it as-is is a silent DEGRADED mode. An
        # UNKNOWN version (a newer format, or garbage) is refused. Only the current version, or
        # one we can migrate forward by RE-DERIVING from the still-present DuckDB file, proceeds.
        migratable = isinstance(version, int) and version in _MIGRATABLE_VERSIONS
        if version != CURRENT_MODEL_VERSION and not migratable:
            _log.warning(
                "refusing dataset %s: semantic model version %r is not supported (expected %d)",
                sidecar.name,
                version,
                CURRENT_MODEL_VERSION,
            )
            continue
        try:
            dataset = Dataset.model_validate(raw)
        except ValueError:
            _log.warning("skipping malformed dataset sidecar %s", sidecar.name)
            continue
        # Existence is not identity: verify the sidecar actually DESCRIBES this data file
        # (row_count + column set) before trusting it, so a swapped/rebuilt table underneath
        # cannot rehydrate a model that is silently wrong about the data.
        mismatch = _identity_mismatch(dataset)
        if mismatch is not None:
            _log.warning("refusing %s: %s", sidecar.name, mismatch)
            continue
        if migratable:
            # Re-derive the fields the newer binder needs (categorical_values is a SELECT
            # DISTINCT over the on-disk data, no LLM) and upgrade the sidecar, so an old dataset
            # gains the fix WITHOUT a re-upload rather than degrading silently.
            _migrate_in_place(dataset)
        store.put(dataset)
        loaded += 1
    return loaded
