"""Evidence packs — the deterministic input to role assignment.

For each column we assemble what the profiler already measured (type, null rate,
cardinality, uniqueness, sign structure, top values, samples) plus the functional
dependencies it participates in, with strength AND support. The column *name* is
included but it is only one weak signal among these; the eval strips headers to prove
the system does not lean on it. No LLM is involved in building this — it is pure
projection over the Profile.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..ingestion.profile import Profile


class DependencyEdge(BaseModel):
    other: str
    strength: float
    support: int


class ColumnEvidence(BaseModel):
    name: str
    semantic_type: str
    null_rate: float
    distinct_count: int
    uniqueness_ratio: float
    min_value: str | None
    max_value: str | None
    integer_ratio: float | None = None
    sign_mix: dict[str, int] | None = None  # negatives / zeros / positives
    top_values: list[dict[str, object]] = Field(default_factory=list)
    samples: list[str] = Field(default_factory=list)
    determines: list[DependencyEdge] = Field(default_factory=list)  # this -> other
    determined_by: list[DependencyEdge] = Field(default_factory=list)  # other -> this


class DatasetEvidence(BaseModel):
    dataset_id: str
    row_count: int
    columns: list[ColumnEvidence]


def build_evidence(profile: Profile) -> DatasetEvidence:
    determines: dict[str, list[DependencyEdge]] = {c.name: [] for c in profile.columns}
    determined_by: dict[str, list[DependencyEdge]] = {c.name: [] for c in profile.columns}
    for fd in profile.functional_dependencies:
        determines[fd.determinant].append(
            DependencyEdge(other=fd.dependent, strength=fd.strength, support=fd.support)
        )
        determined_by[fd.dependent].append(
            DependencyEdge(other=fd.determinant, strength=fd.strength, support=fd.support)
        )

    columns: list[ColumnEvidence] = []
    for col in profile.columns:
        sign_mix = None
        integer_ratio = None
        if col.numeric is not None:
            sign_mix = {
                "negatives": col.numeric.negatives,
                "zeros": col.numeric.zeros,
                "positives": col.numeric.positives,
            }
            integer_ratio = col.numeric.integer_ratio
        columns.append(
            ColumnEvidence(
                name=col.name,
                semantic_type=col.semantic_type,
                null_rate=col.null_rate,
                distinct_count=col.distinct_count,
                uniqueness_ratio=col.uniqueness_ratio,
                min_value=col.min_value,
                max_value=col.max_value,
                integer_ratio=integer_ratio,
                sign_mix=sign_mix,
                top_values=[{"value": v.value, "count": v.count} for v in col.top_values],
                samples=col.samples,
                determines=determines[col.name],
                determined_by=determined_by[col.name],
            )
        )
    return DatasetEvidence(
        dataset_id=profile.dataset_id, row_count=profile.row_count, columns=columns
    )
