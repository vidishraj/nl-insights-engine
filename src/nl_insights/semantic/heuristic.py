"""Deterministic heuristic role proposer — the no-LLM fallback.

The semantic layer must not DEPEND on an LLM: without a key (or a recorded fixture) the
system should degrade, not die. The profiler already computes everything a first guess
needs — types, cardinality, uniqueness, sign mix, integer-dominance, and functional
dependencies — so we can propose obvious roles from that evidence and let the SAME
verifiers test them exactly as they test the model's proposals. 'Heuristics propose,
code disposes' is the identical architecture to 'the model proposes, code disposes':
the LLM becomes a quality improvement on proposal, not a hard dependency.

Every claim it makes is marked ``source='heuristic'`` so provenance stays visible. The
rules read only statistics — never a column name or value — so this stays dataset-
agnostic and passes the anti-hardcoding lint.
"""

from __future__ import annotations

from .evidence import ColumnEvidence, DatasetEvidence
from .model import RoleClaim
from .ontology import EntityKind, Role

_NUMERIC = {"integer", "decimal"}
_TEXTISH = {"text", "unknown"}


def _negatives(col: ColumnEvidence) -> int:
    return (col.sign_mix or {}).get("negatives", 0)


def _int_ratio(col: ColumnEvidence, default: float) -> float:
    # integer_ratio is None for non-numeric columns; 0.0 (a pure decimal) is a REAL value
    # and must not be coalesced away — `x or default` would wrongly treat 0.0 as missing.
    return col.integer_ratio if col.integer_ratio is not None else default


def _looks_like_id(col: ColumnEvidence) -> bool:
    """A positive integer with many distinct values is an identifier, not a measure —
    binding it as a rate/quantity would compute nonsense (revenue = quantity times an id).
    Measures are either decimal (prices/amounts) or lower-cardinality integers (counts,
    which also routinely go negative for returns); an id is neither."""
    return _int_ratio(col, 0.0) >= 0.9 and _negatives(col) == 0 and col.distinct_count >= 1000


def propose_heuristically(evidence: DatasetEvidence) -> list[RoleClaim]:
    cols = evidence.columns
    assigned: dict[str, RoleClaim] = {}

    def claim(
        col: str, role: Role, *, entity: EntityKind | None = None, conf: float, reason: str
    ) -> None:
        assigned[col] = RoleClaim(
            column=col,
            role=role,
            entity=entity,
            confidence=conf,
            reasons=[reason],
            source="heuristic",
        )

    # 1. event_time — the date/timestamp columns. Prefer the one the most other columns
    #    are derived from (it will functionally determine year/month/... period columns).
    times = sorted(
        (c for c in cols if c.semantic_type in {"date", "timestamp"}),
        key=lambda c: len(c.determines),
        reverse=True,
    )
    event_time = times[0].name if times else None
    for c in times:
        claim(c.name, Role.EVENT_TIME, conf=0.7, reason=f"{c.semantic_type} column")

    # 2. period columns — anything the event_time functionally determines is a derived
    #    slice (year/quarter/month/hour), not a measure. Assign before numerics so an
    #    integer 'year' is never mistaken for a quantity. A column that ALSO determines
    #    the event_time (a 1:1 relationship) is key-like, not a slice, so it is excluded
    #    and left for the transaction_key step.
    if event_time is not None:
        for c in cols:
            if c.name in assigned:
                continue
            derived_from_time = any(e.other == event_time for e in c.determined_by)
            determines_time = any(e.other == event_time for e in c.determines)
            if derived_from_time and not determines_time:
                claim(
                    c.name,
                    Role.DIMENSION,
                    conf=0.5,
                    reason="functionally derived from the event_time",
                )

    # 3. flags — boolean or two-valued columns. Assigned BEFORE numerics so a 0/1
    #    indicator is never mistaken for a quantity.
    for c in cols:
        if c.name not in assigned and (c.semantic_type == "boolean" or c.distinct_count == 2):
            claim(c.name, Role.FLAG, conf=0.5, reason="boolean / two-valued")

    # 4. transaction_key — a recurring (non-unique) column that DETERMINES the event_time
    #    (rows sharing it share a time). Most key-like = highest uniqueness among those.
    if event_time is not None:
        keys = sorted(
            (
                c
                for c in cols
                if c.name not in assigned
                and c.uniqueness_ratio < 0.999
                and any(e.other == event_time for e in c.determines)
            ),
            key=lambda c: c.uniqueness_ratio,
            reverse=True,
        )
        if keys:
            claim(
                keys[0].name,
                Role.TRANSACTION_KEY,
                conf=0.6,
                reason="recurs and functionally determines the event_time",
            )

    # 5. numeric MEASURES over what remains (flags and period columns already taken).
    #    An identifier (high-cardinality positive integer) is excluded — binding it as a
    #    rate would compute nonsense revenue. An integer-dominant column is a quantity.
    measures = [
        c
        for c in cols
        if c.name not in assigned
        and c.semantic_type in _NUMERIC
        and c.distinct_count > 2
        and not _looks_like_id(c)
    ]
    qty = next((c for c in measures if _int_ratio(c, 0.0) >= 0.9), None)
    if qty is not None:
        claim(qty.name, Role.ADDITIVE_QUANTITY, conf=0.55, reason="integer-dominant numeric")

    # Money columns. A per-unit RATE and a stored extended AMOUNT are statistically
    # identical — both decimal-bearing — so telling a price from a discount from a shipping
    # fee is a NAMING question, and this proposer reads only statistics. It therefore commits
    # a money role only when the choice is not a coin flip:
    #   • exactly one decimal candidate  → it is unambiguously the per-unit rate;
    #   • exactly two, with a quantity    → a candidate (quantity, rate, amount) triple,
    #     proposed so the stored-amount identity (amount == quantity × rate) can CONFIRM which
    #     decimal is the extended amount; revenue then comes from the verified stored amount.
    # With three or more decimal candidates (or two and no quantity) nothing in the statistics
    # corroborates a choice, so it ABSTAINS: rate/amount stay unbound, net_revenue never binds,
    # and a revenue question REFUSES rather than computing from a guessed column. Declining a
    # question a name alone could answer — but statistics cannot — is the whole thesis.
    decimals = [c for c in measures if c.name not in assigned and _int_ratio(c, 1.0) < 0.9]
    if len(decimals) == 1:
        claim(
            decimals[0].name,
            Role.MONETARY_RATE,
            conf=0.5,
            reason="the only decimal-bearing numeric — an unambiguous per-unit rate",
        )
    elif len(decimals) == 2 and qty is not None:
        claim(
            decimals[0].name,
            Role.MONETARY_RATE,
            conf=0.45,
            reason="candidate per-unit rate in a (quantity, rate, amount) triple",
        )
        claim(
            decimals[1].name,
            Role.MONETARY_AMOUNT,
            conf=0.4,
            reason="candidate stored amount — the stored-amount identity confirms or drops it",
        )

    # 6. text (and any leftover) split into entity-key / description / dimension / ignore
    #    by cardinality and functional-dependency links.
    for c in cols:
        if c.name in assigned:
            continue
        if c.semantic_type not in _TEXTISH:  # a leftover numeric with no measure fit
            claim(c.name, Role.DIMENSION, conf=0.3, reason="categorical")
            continue
        if c.uniqueness_ratio >= 0.9:  # near-unique free text
            role = Role.DESCRIPTION if (c.determines or c.determined_by) else Role.IGNORE
            claim(c.name, role, conf=0.35, reason="near-unique free text")
            continue
        if c.distinct_count >= 20:  # many repeated values → an identifier
            claim(
                c.name,
                Role.ENTITY_KEY,
                entity=EntityKind.OTHER,
                conf=0.4,
                reason="repeated identifier-like text (kind unknown without a name)",
            )
            continue
        claim(c.name, Role.DIMENSION, conf=0.4, reason="low-cardinality categorical")

    return list(assigned.values())
