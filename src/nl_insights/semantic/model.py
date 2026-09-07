"""The Semantic Model — the system's machine-checked understanding of a dataset.

It is versioned and evidence-carrying. It is the *sole* input to
the interpreter: the interpreter never sees raw data or raw column names, only bound
roles it can reason over. Every binding records the LLM's claim (role, confidence,
reasons) AND the deterministic verifier results that tested it, so a reader can see
not just what the system decided but why it is entitled to believe it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .ontology import EntityKind, Role


class VerificationStatus(StrEnum):
    """Absence of evidence and evidence of absence are different things."""

    VERIFIED = "verified"  # a verifier tested the claim and it held
    UNVERIFIED = "unverified"  # no applicable test — the claim stands unchallenged
    REFUTED = "refuted"  # a verifier tested the claim and it FAILED — do not trust it


class RoleClaim(BaseModel):
    """A proposed role for one column — from the LLM, or from the heuristic fallback.

    ``source`` records WHICH proposer made the claim so the model stays honest about
    provenance (an LLM binding and a heuristic binding are both verified the same way,
    but a reader/UI should be able to tell them apart)."""

    column: str
    role: Role
    entity: EntityKind | None = None  # only for entity_key
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    source: str = "llm"  # 'llm' | 'heuristic'


# CONFIRMING verifiers prove a claim when they PASS but do not disprove it when they FAIL —
# their failure is "could not confirm" (UNVERIFIED), never "disproved" (REFUTED). The
# stored-amount identity is confirming: amount == quantity × rate is SUFFICIENT evidence a
# column is an extended amount, but a column that fails it (e.g. a total carrying tax, so
# total = quantity × rate × 1.05) is still an amount — the identity simply does not apply.
# Refutation requires POSITIVE contrary evidence from a DISQUALIFYING test (a non-parsing
# date bound as event_time, a non-numeric column bound as a quantity) — never mere absence of
# proof. This is the rule additive_quantity_structure and monetary_rate_structure already
# follow ("a weak signal, never a disproof"); the amount identity was the lone violator.
_CONFIRMING_VERIFIERS = frozenset({"stored_amount_identity"})


class VerifierResult(BaseModel):
    """One deterministic (SQL) test of a claim or a discovered structural fact."""

    name: str
    passed: bool
    detail: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ColumnBinding(BaseModel):
    """A column's final role, its claim, and the verifiers that tested it."""

    column: str
    role: Role
    entity: EntityKind | None = None
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    verifiers: list[VerifierResult] = Field(default_factory=list)
    provenance: str = "llm"  # where the role claim came from: 'llm' | 'heuristic'

    @property
    def status(self) -> VerificationStatus:
        # A DISQUALIFYING test that failed refutes; a CONFIRMING test that failed does not
        # (see _CONFIRMING_VERIFIERS). A single passing test verifies; nothing applicable
        # leaves the claim UNVERIFIED — standing, unchallenged, still usable with disclosure.
        if any(not v.passed and v.name not in _CONFIRMING_VERIFIERS for v in self.verifiers):
            return VerificationStatus.REFUTED
        if any(v.passed for v in self.verifiers):
            return VerificationStatus.VERIFIED
        return VerificationStatus.UNVERIFIED

    @property
    def refuted(self) -> bool:
        return self.status is VerificationStatus.REFUTED


class ReturnsConvention(BaseModel):
    """How the data marks a return/cancellation — discovered, never hardcoded."""

    kind: str  # explicit_flag | explicit_category | derived_negative_quantity | none
    column: str | None = None
    return_values: list[str] = Field(default_factory=list)
    detail: str = ""
    confidence: float = 0.0


class PartitionDimension(BaseModel):
    """A categorical column that partitions the fact table.

    e.g. a line-type column where only one value is a real product line: 'top products
    by revenue' is wrong unless non-product rows are excluded. Discovered from a flag
    it (nearly) determines, so it is never hardcoded and works on any such file.
    """

    column: str
    fact_value: str | None = None  # the value that selects the fact rows (e.g. product lines)
    via_flag: str | None = None  # the flag column it was correlated against
    detail: str = ""


class EntityLabel(BaseModel):
    """A human-readable LABEL for an entity — a description column functionally tied to
    an entity key (e.g. a product description <-> product code).

    Discovered from an FD in either direction, never from a name. It matters because a
    human ranks 'top products' by their *description*, not their opaque code: grouping
    by the label is grouping the entity, so it must inherit the entity's fact-table
    treatment (the product-line partition). Missing this returns postage as a product.
    """

    column: str  # the label column (a description)
    entity_column: str  # the entity key it labels
    entity: EntityKind | None = None
    strength: float
    support: int
    detail: str = ""


class PeriodCompleteness(BaseModel):
    """A flag marking whether a period is complete (e.g. a partial first quarter).

    Period-over-period measures must exclude incomplete periods or they are wrong.
    """

    flag_column: str
    period_column: str | None = None
    complete_value: str = "true"
    detail: str = ""


class CategoricalValues(BaseModel):
    """The observed values of a categorical column, so a filter VALUE can be validated the
    way a column NAME already is. ``values`` is a top-N sample; ``exhaustive`` is True only
    when that sample is provably ALL of them (distinct_count did not exceed the sample size).
    A filter value absent from an EXHAUSTIVE set certainly does not exist; absence from a
    non-exhaustive sample proves nothing, so it is never grounds to refuse."""

    values: list[str]
    exhaustive: bool


class MeasureBinding(BaseModel):
    """A declarative measure resolved over ROLES, with a SQL expression if available."""

    name: str
    expression: str  # human-readable, over roles
    grain: str  # row | transaction | arbitrary
    additive: bool
    available: bool  # are the required roles bound?
    sql: str | None = None  # concrete aggregate expression when available
    missing_roles: list[Role] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SemanticModel(BaseModel):
    # Bump this whenever the schema gains a field the BINDER relies on, so a sidecar written by
    # an older version of the code is not silently loaded as "compatible" while missing data the
    # binder now depends on (that is a silent DEGRADED mode — the very failure this project
    # argues against). v2 added `categorical_values` (the binder validates filter VALUES);
    # v3 added `numeric_columns` (the binder validates generic avg/sum/min/max). An older
    # sidecar is re-derived on load (see jobs.store), not trusted as-is.
    version: int = 3
    dataset_id: str
    table: str
    row_count: int = 0
    bindings: list[ColumnBinding]
    returns: ReturnsConvention
    partition_dimensions: list[PartitionDimension] = Field(default_factory=list)
    labels: list[EntityLabel] = Field(default_factory=list)
    period_completeness: list[PeriodCompleteness] = Field(default_factory=list)
    measures: list[MeasureBinding] = Field(default_factory=list)
    coverage: dict[str, float] = Field(default_factory=dict)  # column -> non-null coverage
    # column -> its observed values, so the binder can validate a filter VALUE (not just the
    # column name). Populated for low-cardinality categoricals where the sample is exhaustive.
    categorical_values: dict[str, CategoricalValues] = Field(default_factory=dict)
    # columns whose STORED TYPE is numeric (integer/decimal), regardless of the ROLE the
    # proposer gave them — so a generic aggregation (avg/sum/min/max) can run over an ordinary
    # numeric the retail measure algebra does not name (a fare, a life expectancy, a price the
    # proposer called 'ignore'). The binder validates avg/sum/min/max against THIS set. Added
    # in v3; an older sidecar re-derives it from the DuckDB column types on load.
    numeric_columns: list[str] = Field(default_factory=list)
    # date column -> a human caveat about how its day/month order was resolved, so a
    # time query can DISCLOSE the interpretation instead of silently trusting a coin flip.
    date_assumptions: dict[str, str] = Field(default_factory=dict)
    # event_time columns stored as TEXT that parse to a timestamp -> the SQL EXPRESSION
    # the executor must emit (a try_strptime with the DETECTED order, or a TRY_CAST),
    # instead of comparing raw strings and silently returning nothing.
    temporal_cast_sql: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def columns_in_role(self, role: Role, *, include_refuted: bool = False) -> list[ColumnBinding]:
        """Bindings in a role. Refuted bindings are excluded by default — a role a
        verifier disproved must not be reachable as if it were sound."""
        return [b for b in self.bindings if b.role == role and (include_refuted or not b.refuted)]

    def first_in_role(self, role: Role, *, include_refuted: bool = False) -> ColumnBinding | None:
        found = self.columns_in_role(role, include_refuted=include_refuted)
        return found[0] if found else None

    def entity_key_columns(self, entity: EntityKind) -> set[str]:
        return {
            b.column
            for b in self.bindings
            if b.role is Role.ENTITY_KEY and b.entity is entity and not b.refuted
        }

    def product_identifier_columns(self) -> set[str]:
        """Every column that identifies a PRODUCT: the product entity key(s) plus any
        label (description) FD-linked to one. Ranking/grouping by ANY of these is a
        product ranking, so the product-line partition must apply — whether the human
        grouped by the opaque code or by the description. This is why the trap ('top
        products by revenue' returning postage) closes for both."""
        keys = self.entity_key_columns(EntityKind.PRODUCT)
        labels = {lbl.column for lbl in self.labels if lbl.entity is EntityKind.PRODUCT}
        return keys | labels
