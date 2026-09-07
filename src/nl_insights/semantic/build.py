"""Build a verified Semantic Model: profile + data + provider -> SemanticModel.

The flow is the architecture in miniature: the LLM PROPOSES roles over the evidence,
then deterministic SQL DISPOSES — testing each testable claim, discovering the returns
convention and the partition/period structure, and resolving the measure algebra. The
model that comes out carries every claim, its confidence, and the verifier results, so
it is fully inspectable.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb

from ..ingestion.dates import DateOrder, DateResolution
from ..ingestion.profile import ColumnProfile, Profile
from ..provider import LLMProvider, ReplayCacheMiss
from . import verifiers
from .evidence import build_evidence
from .heuristic import propose_heuristically
from .measures import build_measures
from .model import CategoricalValues, ColumnBinding, ReturnsConvention, SemanticModel
from .ontology import Role
from .proposer import propose_roles

# Narration/cancellation hook: (stage, message); may raise to abort. Defaults to no-op.
StageFn = Callable[[str, str], None]


def _noop(_stage: str, _message: str) -> None:
    return None


def _date_assumptions(resolutions: dict[str, DateResolution], columns: set[str]) -> dict[str, str]:
    """A human caveat per event_time column whose day/month order was resolved to a
    non-ISO interpretation, so a time query DISCLOSES the interpretation it used."""
    out: dict[str, str] = {}
    for col in columns:
        res = resolutions.get(col)
        if res is None:
            continue
        if res.order is DateOrder.DAY_FIRST:
            out[col] = f"dates in {col!r} read as day-first (DD/MM/YYYY) — {res.evidence}"
        elif res.order is DateOrder.MONTH_FIRST:
            out[col] = f"dates in {col!r} read as month-first (MM/DD/YYYY) — {res.evidence}"
        elif res.order is DateOrder.AMBIGUOUS:
            out[col] = (
                f"dates in {col!r} are ambiguous (nothing settles day vs month); assumed day-first"
            )
    return out


def build_semantic_model(
    con: duckdb.DuckDBPyConnection,
    table: str,
    profile: Profile,
    provider: LLMProvider,
    *,
    model: str = "claude-sonnet-4-5",
    on_stage: StageFn | None = None,
    date_orders: dict[str, DateResolution] | None = None,
) -> SemanticModel:
    emit = on_stage or _noop
    date_orders = date_orders or {}
    by_name: dict[str, ColumnProfile] = {c.name: c for c in profile.columns}
    evidence = build_evidence(profile)
    emit("inferring", "proposing column roles from the evidence (LLM proposes)")
    try:
        claims = propose_roles(provider, evidence, model)
    except ReplayCacheMiss:
        # No recorded LLM response for this (unseen) dataset. Rather than die, fall back
        # to the deterministic heuristic proposer — the semantic layer is not LLM-
        # dependent by construction. The verifiers below test its proposals identically.
        emit(
            "inferring",
            "no recorded LLM response — proposing roles heuristically from the profile",
        )
        claims = propose_heuristically(evidence)

    # Keep the highest-confidence claim per column.
    best: dict[str, ColumnBinding] = {}
    for claim in sorted(claims, key=lambda c: c.confidence, reverse=True):
        best.setdefault(
            claim.column,
            ColumnBinding(
                column=claim.column,
                role=claim.role,
                entity=claim.entity,
                confidence=claim.confidence,
                reasons=claim.reasons,
                provenance=claim.source,
            ),
        )
    bindings = list(best.values())

    # Proposal-level resolutions, used only to WIRE the verifiers (which testable
    # property to check). Whether they survive is decided by the verifiers below.
    def _first_proposed(role: Role) -> str | None:
        found = [b.column for b in bindings if b.role == role]
        return found[0] if found else None

    p_event_time = _first_proposed(Role.EVENT_TIME)
    p_quantity = _first_proposed(Role.ADDITIVE_QUANTITY)
    p_rate = _first_proposed(Role.MONETARY_RATE)

    # --- verify claims (SQL where testable, profile stats otherwise) ---
    emit("verifying", "testing each testable claim against the data (code disposes)")
    proposed_cast: dict[str, str] = {}  # event_time column -> parse expression, if any
    for b in bindings:
        col = by_name.get(b.column)
        if b.role == Role.EVENT_TIME and col:
            # an event_time must be a REAL timestamp, not just present. Probe with the
            # parser the system already has (the detected date order) — a non-parsing
            # VARCHAR is refuted so time queries refuse instead of returning silent nulls.
            result, cast_sql = verifiers.event_time_temporal(
                con, table, col, date_orders.get(b.column)
            )
            b.verifiers.append(result)
            if cast_sql:
                proposed_cast[b.column] = cast_sql
        elif b.role == Role.TRANSACTION_KEY and p_event_time:
            b.verifiers.append(
                verifiers.transaction_key_shares_time(con, table, b.column, p_event_time)
            )
        elif b.role == Role.ADDITIVE_QUANTITY and col:
            b.verifiers.append(verifiers.additive_quantity_structure(col))
        elif b.role == Role.MONETARY_RATE and col:
            b.verifiers.append(verifiers.monetary_rate_structure(col))
        elif b.role == Role.ENTITY_KEY and col:
            b.verifiers.append(verifiers.entity_coverage(col))
        if b.role == Role.MONETARY_AMOUNT and p_quantity and p_rate:
            b.verifiers.append(
                verifiers.stored_amount_identity(con, table, p_quantity, p_rate, b.column)
            )

    # --- disposal: a refuted claim is demoted, not merely annotated. Its confidence
    #     collapses and it becomes unreachable to role resolution and the measures. ---
    for b in bindings:
        if b.refuted:
            b.confidence = 0.0
            b.reasons = [*b.reasons, "REFUTED by a verifier — role dropped from resolution"]

    def cols_for(role: Role) -> list[str]:  # non-refuted only
        return [b.column for b in bindings if b.role == role and not b.refuted]

    def first(role: Role) -> str | None:
        found = cols_for(role)
        return found[0] if found else None

    quantity = first(Role.ADDITIVE_QUANTITY)
    key = first(Role.TRANSACTION_KEY)
    event_time = first(Role.EVENT_TIME)
    roles = {b.column: b.role for b in bindings if not b.refuted}
    entities = {b.column: b.entity for b in bindings if not b.refuted}
    # The SET of amount columns that themselves passed the stored-amount identity — not a
    # scalar any(). measures.py selects the revenue amount from this set explicitly, so a
    # 'verified stored amount' label is only ever earned by the column that verified, never
    # by a different amount column that merely happened to be first (the false-provenance bug).
    verified_amounts = {
        b.column
        for b in bindings
        if b.role == Role.MONETARY_AMOUNT
        and not b.refuted
        and any(v.name == "stored_amount_identity" and v.passed for v in b.verifiers)
    }

    # --- discover conventions and structure (deterministic) ---
    emit("discovering", "discovering returns, partitions, labels and period structure")
    if quantity:
        returns = verifiers.discover_returns(
            con,
            table,
            profile,
            quantity=quantity,
            transaction_key=key,
            flags=cols_for(Role.FLAG),
            dimensions=cols_for(Role.DIMENSION),
        )
    else:
        returns = ReturnsConvention(kind="none", detail="no additive_quantity bound")
    periods = verifiers.period_columns(profile, event_time, roles)
    partition_dimensions = verifiers.discover_partition_dimensions(
        con, table, profile, roles, periods
    )
    period_completeness = verifiers.discover_period_completeness(profile, roles, periods)
    labels = verifiers.discover_entity_labels(profile, roles, entities)

    role_columns = {role: cols_for(role) for role in Role}
    measures = build_measures(role_columns, verified_amounts=verified_amounts, returns=returns)

    coverage = {c.name: round(1.0 - c.null_rate, 6) for c in profile.columns}

    # Per-column observed values, so the binder can validate a filter VALUE (not just the
    # column name) against the data — only for BOUND columns, and marked EXHAUSTIVE only when
    # the top-N sample provably captured every distinct value (distinct_count did not exceed
    # it). A value absent from an exhaustive set certainly does not exist.
    categorical_values = {
        c.name: CategoricalValues(
            values=[str(v.value) for v in c.top_values],
            exhaustive=c.distinct_count <= len(c.top_values),
        )
        for c in profile.columns
        if c.name in roles and c.top_values
    }

    # A surviving event_time that is text-but-parseable carries a parse expression; record
    # it so the executor emits it (a try_strptime with the detected order, or a TRY_CAST)
    # rather than comparing raw strings.
    surviving_event_time = {
        b.column for b in bindings if b.role == Role.EVENT_TIME and not b.refuted
    }
    temporal_cast_sql = {c: expr for c, expr in proposed_cast.items() if c in surviving_event_time}
    date_assumptions = _date_assumptions(date_orders, surviving_event_time)

    # Columns whose STORED type is numeric, regardless of role — so a generic aggregation can
    # run over an ordinary numeric the measure algebra does not name. Taken from the profile's
    # own numeric detection (integer/decimal), the same source _attach_numeric_stats uses.
    numeric_columns = [c.name for c in profile.columns if c.numeric is not None]

    return SemanticModel(
        dataset_id=profile.dataset_id,
        table=table,
        row_count=profile.row_count,
        bindings=bindings,
        returns=returns,
        partition_dimensions=partition_dimensions,
        labels=labels,
        period_completeness=period_completeness,
        measures=measures,
        coverage=coverage,
        categorical_values=categorical_values,
        numeric_columns=numeric_columns,
        temporal_cast_sql=temporal_cast_sql,
        date_assumptions=date_assumptions,
    )
