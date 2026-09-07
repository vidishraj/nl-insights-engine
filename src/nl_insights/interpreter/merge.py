"""Follow-up handling as IR merge — we diff plans, we do not replay the conversation.

A follow-up ('and what about last month?') yields a PARTIAL plan; we overlay only the
fields it provides onto the previous turn's IR, so measure and grouping are inherited
while the time filter is overridden. This is deterministic, inspectable, and diffable —
and it means the model never re-reads chat history. A partial that changes nothing, or
that cannot be reconciled, is surfaced (via a note) for the binder to turn into a
clarify verdict rather than being guessed.
"""

from __future__ import annotations

from typing import Any

from .ir import QueryIR

# Only these top-level fields may be overridden by a follow-up. 'intent'/'notes' are
# regenerated, not inherited literally.
_MERGEABLE = {
    "measures",
    "group_by",
    "categorical_filters",
    "numeric_filters",
    "time",
    "top_k",
    "period_comparison",
    "share_of_total",
    "distinct_count_of",
    "frequency",
    "basket",
}


def merge_ir(previous: QueryIR, partial: dict[str, Any]) -> QueryIR:
    """Overlay the follow-up's provided fields onto the previous IR."""
    overrides = {k: v for k, v in partial.items() if k in _MERGEABLE}
    base = previous.model_dump()
    base.update(overrides)
    base["intent"] = str(partial.get("intent") or previous.intent)
    # Diagnostic fields are regenerated per interpretation, never inherited: a follow-up that
    # resolves a previously unmet measure or grouping must not keep refusing/clarifying
    # against the old gap.
    base["unmet_concepts"] = list(partial.get("unmet_concepts") or [])
    base["unmet_dimensions"] = list(partial.get("unmet_dimensions") or [])

    notes = list(previous.notes)
    if not overrides:
        notes.append("follow-up changed nothing from the previous plan — needs clarification")
    changed = sorted(overrides)
    if changed:
        notes.append(f"merged follow-up: overrode {', '.join(changed)}")
    base["notes"] = notes
    return QueryIR.model_validate(base)
