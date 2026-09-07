"""The Query IR — a small typed plan that English compiles to.

This Pydantic model *is* the one-line explanation: the model turns a question into
this JSON, and everything after it is ordinary, steppable code. The IR is written over
ROLES and the semantic model's named measures/dimensions, never over raw SQL. It is
deliberately small but covers the five example questions with headroom (top-k,
period-over-period, share-of-total, distinct counts, per-entity frequency, baskets),
because the live walkthrough asks unseen questions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Grain(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# The operator is a CLOSED set on the schema surface the model reads (a Literal → a JSON
# `enum`), not a free-form string. A bare `str` let the model emit 'not in'/'!='/'neq',
# which the compile sites' `dict.get(op, default)` then silently mapped to the OPPOSITE
# branch — 'revenue excluding cancelled orders' returning exactly the cancelled orders.
# Same lesson as TimeWindow.end: put the contract where the other component reads it. The
# binder additionally refuses any op outside these sets (defence for a programmatically
# constructed IR that bypasses validation), and the compilers index the maps directly so
# an unmapped op raises rather than defaults.
CategoricalOp = Literal["in", "not_in"]
NumericOp = Literal["gt", "gte", "lt", "lte", "eq", "between"]
FrequencyOp = Literal["eq", "lte", "gte"]


class CategoricalFilter(BaseModel):
    column: str
    op: CategoricalOp = "in"
    values: list[str] = Field(default_factory=list)


class NumericFilter(BaseModel):
    column: str
    op: NumericOp
    value: float
    value2: float | None = None  # upper bound for 'between'


class TimeWindow(BaseModel):
    # The contract lives HERE, in the schema the model sees — not only in a comment. For a
    # whole named month or quarter, use named_period; start/end are for arbitrary ranges,
    # start INCLUSIVE and end EXCLUSIVE.
    grain: Grain | None = Field(
        default=None, description="bucket size for grouping or period comparison"
    )
    start: str | None = Field(
        default=None, description="INCLUSIVE lower bound, full ISO date 'YYYY-MM-DD'"
    )
    end: str | None = Field(
        default=None,
        description=(
            "EXCLUSIVE upper bound, full ISO date 'YYYY-MM-DD' — the first day AFTER the "
            "range (a whole month becomes the 1st of the FOLLOWING month, not the last day "
            "of the month). For a whole named month/quarter use named_period instead."
        ),
    )
    last_n: int | None = Field(default=None, description="the last N periods at 'grain'")
    named_period: str | None = Field(
        default=None,
        description=(
            "PREFER this for a whole named period: a month as 'YYYY-MM' or a quarter as "
            "'YYYY-Qn'. Do NOT also set start/end when using it."
        ),
    )


class TopK(BaseModel):
    measure: str  # what to rank by — must resolve to a computed measure or a grouping
    k: int = Field(10, ge=1)  # a rank size is >= 1; a negative/zero LIMIT is not a plan
    direction: str = "desc"  # desc | asc


class PeriodComparison(BaseModel):
    """Compare a measure across two periods (e.g. growth between quarters)."""

    grain: Grain
    period_a: str | None = None  # earlier period label; None = second-most-recent complete
    period_b: str | None = None  # later period label; None = most-recent complete
    kind: Literal["growth", "change", "ratio"] = "growth"


class FrequencyConstraint(BaseModel):
    """Per-entity transaction frequency, e.g. customers who bought exactly once."""

    entity: str  # the entity column (e.g. a customer key)
    count_of: str  # what to count per entity (e.g. the transaction key)
    op: FrequencyOp = "eq"
    n: int = 1


class Aggregation(BaseModel):
    """A GENERIC aggregation over a column, for questions the named measure algebra does not
    cover: counting rows, or averaging/summing/min/max of an ordinary numeric column that is
    not part of a revenue definition (a fare, a life expectancy, a diamond price). This is
    what lets an unfamiliar, non-retail file still answer 'how many rows' and 'average X'."""

    func: Literal["count", "sum", "avg", "min", "max"]
    # the column to aggregate. None (only valid for func='count') means count(*) — the ROW
    # count, which needs no column and always applies. For sum/avg/min/max the column must be
    # numeric; the binder refuses otherwise, and refuses count(*) written as sum/avg/etc.
    column: str | None = None


class QueryIR(BaseModel):
    """A typed analytical plan proposed by the interpreter, disposed by the binder."""

    intent: str = ""  # a short paraphrase of the question, for the explanation
    measures: list[str] = Field(default_factory=list)  # names from the measure algebra
    group_by: list[str] = Field(default_factory=list)  # dimension columns
    categorical_filters: list[CategoricalFilter] = Field(default_factory=list)
    numeric_filters: list[NumericFilter] = Field(default_factory=list)
    time: TimeWindow | None = None
    top_k: TopK | None = None
    period_comparison: PeriodComparison | None = None
    share_of_total: bool = False
    distinct_count_of: str | None = None  # count distinct of a column (unique customers/orders)
    frequency: FrequencyConstraint | None = None
    basket: bool = False  # products bought together within a transaction
    # generic aggregations (count(*), avg/sum/min/max of a numeric column) for questions the
    # named measure algebra does not cover — the path that lets a non-retail file answer.
    aggregations: list[Aggregation] = Field(default_factory=list)
    # concepts the question asked for that the model CANNOT express (a quantity that needs
    # data this file does not carry). Naming them here — instead of substituting a different
    # measure — is what lets the binder refuse honestly rather than invent an answer.
    unmet_concepts: list[str] = Field(
        default_factory=list,
        description=(
            "measures/concepts the question asked for that are NOT in the semantic model "
            "(a quantity for which the model lists no columns). List them here and do NOT "
            "substitute a different measure — a substituted answer to a different question "
            "is wrong."
        ),
    )
    # attributes the question asked to GROUP BY (break down by) or FILTER ON that are not
    # available columns/entities in this dataset. Surfacing them here — instead of silently
    # dropping the grouping to return a bare total, or substituting a different column — is
    # what lets the binder CLARIFY rather than answer a differently-shaped question.
    unmet_dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "attributes the question asked to break down BY or filter ON that are NOT "
            "available dimension/entity columns (an attribute this dataset has no column for). "
            "List the user's term here and do NOT drop the grouping to return an ungrouped "
            "total or substitute another column — a breakdown that silently collapses to a "
            "scalar, or regroups by a different dimension, is a wrong answer to a different "
            "question."
        ),
    )
    notes: list[str] = Field(default_factory=list)  # incl. parts the IR cannot express
