"""The Answer artifact — value + plan + SQL + a plan-derived explanation.

The explanation is RENDERED FROM THE EXECUTED PLAN, never asked of a model, so it
cannot lie about what was computed: it states the formula, the filters, the grain, and
the coverage that actually ran. The Answer is fully serialisable and reproducible from
(plan + data).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..binder.verdict import BoundPlan


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
