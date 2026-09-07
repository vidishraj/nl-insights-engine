"""Ingestion — CSV to a queryable DuckDB table plus a deterministic Profile.

SQL only, no LLM. Owns dialect sniffing, full-file type validation, evidence-based
date-order resolution, and the profile (types, null rates, cardinalities, sign mixes,
top values, samples, and functional dependencies) the semantic layer reasons over.
"""

from .dates import DateOrder, DateResolution, resolve_date_order
from .dialect import Dialect, DialectError, sniff_dialect
from .loader import IngestionError, IngestResult, ingest
from .profile import (
    ColumnProfile,
    FunctionalDependency,
    NumericStats,
    Profile,
    ValueCount,
    compute_profile,
)

__all__ = [
    "ColumnProfile",
    "DateOrder",
    "DateResolution",
    "Dialect",
    "DialectError",
    "FunctionalDependency",
    "IngestResult",
    "IngestionError",
    "NumericStats",
    "Profile",
    "ValueCount",
    "compute_profile",
    "ingest",
    "resolve_date_order",
    "sniff_dialect",
]
