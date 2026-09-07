"""Verdict types — the binder's output, and the one place refusal is decided.

Refusal is structural, not a matter of the model choosing to be honest: the binder
validates the typed plan against the semantic model and emits exactly one verdict.
Every REFUSE/CLARIFY carries evidence from the model (missing roles, coverage numbers),
not prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..interpreter.ir import CategoricalFilter, QueryIR


class VerdictKind(StrEnum):
    ANSWERABLE = "answerable"
    REFUSE = "refuse"
    CLARIFY = "clarify"
    ANSWER_WITH_CAVEATS = "answer_with_caveats"


class Caveat(BaseModel):
    kind: str  # assumption | partition | period | coverage | degenerate
    detail: str
    metrics: dict[str, float] = Field(default_factory=dict)


class BoundMeasure(BaseModel):
    name: str
    sql: str


class BoundPlan(BaseModel):
    """A validated, reference-resolved plan the executor can compile deterministically."""

    ir: QueryIR
    table: str
    measures: list[BoundMeasure]
    group_by: list[str]
    applied_filters: list[CategoricalFilter] = Field(default_factory=list)  # binder defaults
    require_complete_period_flag: str | None = None  # exclude incomplete periods
    coverage: dict[str, float] = Field(default_factory=dict)


class Verdict(BaseModel):
    kind: VerdictKind
    plan: BoundPlan | None = None
    reason: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    clarify_question: str | None = None
    options: list[str] = Field(default_factory=list)
    caveats: list[Caveat] = Field(default_factory=list)
