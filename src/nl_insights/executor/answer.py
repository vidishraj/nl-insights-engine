"""The Answer artifact — value + plan + SQL + a plan-derived explanation.

The explanation is RENDERED FROM THE EXECUTED PLAN, never asked of a model, so it
cannot lie about what was computed: it states the formula, the filters, the grain, and
the coverage that actually ran. The Answer is fully serialisable and reproducible from
(plan + data).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..binder.verdict import BoundPlan, VerdictKind


class Answer(BaseModel):
    rows: list[dict[str, Any]]
    columns: list[str]
    sql: str
    explanation: str  # deterministic, rendered from the plan
    formula: list[str]  # each measure's formula over roles
    filters_applied: list[str]  # human-readable, including binder defaults
    grain: str
    assumptions: list[str]  # named assumptions (e.g. revenue is net of returns)
    caveats: list[str]
    coverage: dict[str, float]
    tier: str = "planned"  # 'planned' (semantic path) — the guarded raw-SQL tier is separate
    plan: BoundPlan

    def verdict_kind(self) -> VerdictKind:
        """The verdict CLASS a finished answer earns - THE single place the class is decided for
        an answered query, so every consumer reports the same one. A verdict is a claim about the
        answer, and the answer does not exist until execute; a class computed at bind time is
        computed before the thing it describes exists and cannot see an execute-time caveat (the
        non-fact partition disclosure, for one). So it is finalised HERE, from the finished answer.

        ANSWER_WITH_CAVEATS iff a real (non-assumption) caveat is present. The ``caveats`` field
        already holds exactly the non-assumption disclosures (assumptions live in ``assumptions``),
        so a non-empty ``caveats`` is precisely 'there is a real limitation'.
        """
        return VerdictKind.ANSWER_WITH_CAVEATS if self.caveats else VerdictKind.ANSWERABLE
