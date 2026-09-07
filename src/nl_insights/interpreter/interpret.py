"""NL -> Query IR — the second (and only question-facing) LLM call.

The interpreter sees only the SEMANTIC MODEL, never raw data. It is given the named
measures, the bound dimensions, and the time/returns structure, and it emits a typed
QueryIR via forced structured output — never SQL, never free text. Whatever it
proposes is then handed to the binder, which decides answerable / refuse / clarify.
"""

from __future__ import annotations

from typing import Any

from ..provider import LLMProvider, LLMRequest
from ..semantic.model import SemanticModel
from ..semantic.ontology import Role
from .ir import QueryIR

SYSTEM = (
    "You translate a business question into a small typed query plan (the schema you "
    "are given). Use ONLY the measures and dimensions named in the provided semantic "
    "model; never invent a column or write SQL. Reference measures by their name and "
    "dimensions/entities by their column. Prefer the measure algebra over raw columns. "
    "Measure NAMES are generic (revenue/units/orders) and often will NOT match this "
    "dataset's domain; each measure lists the actual 'columns' it computes over, which carry "
    "the dataset's own words. Map the user's term to the measure whose columns MEAN what the "
    "user means, EVEN WHEN the user's wording differs from the column's name: the user names a "
    "CONCEPT, not necessarily a column, so a measure summing (say) a 'haulage_outlay' column "
    "answers 'total shipping spend'. Do "
    "not refuse a computable question merely because its wording is not the generic noun. "
    "If the question's "
    "core MEASURE is not one the model offers (it names a quantity for which the model lists "
    "no columns), do NOT substitute a different measure — list the missing concept "
    "in 'unmet_concepts' and leave measures empty; a substituted answer is wrong. If the "
    "question asks to GROUP BY or FILTER ON an attribute that is not an available dimension/"
    "entity column (an attribute this dataset has no column for), "
    "but the MEASURE itself is computable, do NOT drop the "
    "grouping and return an ungrouped total, and do NOT substitute a different column — list "
    "the user's term in 'unmet_dimensions'; a requested breakdown that silently becomes a "
    "scalar, or regroups by a different dimension, is a wrong answer. (But if the PRIMARY thing "
    "the question asks to count or measure does not exist at all — an entity this dataset does "
    "not record — that is an unmet CONCEPT: put it in 'unmet_concepts', not "
    "'unmet_dimensions'.) For a plain COUNT of rows ('how many X are there') use "
    "aggregations=[{func:'count'}] (no column) — it always applies. To summarise an ordinary "
    "NUMERIC column that is not a named measure (an average, min, max, or sum of a value listed "
    "in numeric_columns), use aggregations=[{func:'avg'|'sum'|'min'|'max', column:<that column>}]; "
    "prefer a named measure when one fits, and use a generic aggregation for the rest. "
    "Only genuinely cosmetic parts you cannot express go in 'notes'. For a "
    "whole named month or quarter use time.named_period (a month as 'YYYY-MM', a quarter as "
    "'YYYY-Qn'), NOT start/end; start is inclusive and end is EXCLUSIVE. Return the plan and "
    "nothing else."
)


def semantic_context(model: SemanticModel) -> dict[str, Any]:
    """A compact, interpretation-facing view of the semantic model for the prompt."""
    dims = [b.column for b in model.bindings if b.role == Role.DIMENSION]
    entities = [
        {"column": b.column, "entity": b.entity}
        for b in model.bindings
        if b.role == Role.ENTITY_KEY
    ]
    event_time = model.first_in_role(Role.EVENT_TIME)
    # Role -> bound columns, so each measure can be shown with the CONCRETE columns it
    # aggregates. The measure NAMES are generic nouns (net_revenue, units_sold …); the columns
    # carry the dataset's own domain words, which is what lets the model map a domain term to
    # the right measure without any dataset vocabulary being baked into the prompt.
    role_columns: dict[str, list[str]] = {}
    for b in model.bindings:
        if not b.refuted:
            role_columns.setdefault(b.role.value, []).append(b.column)

    def _measure_columns(expression: str) -> list[str]:
        cols: list[str] = []
        for role_value, columns in role_columns.items():
            if role_value in expression:
                cols.extend(columns)
        return cols

    return {
        "measures": [
            {
                "name": m.name,
                "expression": m.expression,
                "columns": _measure_columns(m.expression),
                "available": m.available,
            }
            for m in model.measures
        ],
        "dimensions": dims,
        "entities": entities,
        # numeric columns available for a GENERIC aggregation (avg/sum/min/max) even if they
        # are not part of a named measure — this is what lets an ordinary numeric be summarised.
        "numeric_columns": list(model.numeric_columns),
        "event_time": event_time.column if event_time else None,
        "time_grains": ["day", "week", "month", "quarter", "year"] if event_time else [],
        "returns_convention": model.returns.kind,
        "partition_defaults": [
            {"column": p.column, "fact_value": p.fact_value} for p in model.partition_dimensions
        ],
        "period_completeness": [p.flag_column for p in model.period_completeness],
    }


def _schema() -> dict[str, Any]:
    return QueryIR.model_json_schema()


def build_request(model: SemanticModel, question: str, llm_model: str) -> LLMRequest:
    import json

    prompt = (
        f"Semantic model:\n{json.dumps(semantic_context(model))}\n\n"
        f"Question: {question}\n\nEmit the query plan."
    )
    return LLMRequest(model=llm_model, system=SYSTEM, prompt=prompt, schema=_schema())


def build_followup_request(
    model: SemanticModel, previous: QueryIR, question: str, llm_model: str
) -> LLMRequest:
    import json

    prompt = (
        f"Semantic model:\n{json.dumps(semantic_context(model))}\n\n"
        f"Previous plan:\n{previous.model_dump_json()}\n\n"
        f"Follow-up question: {question}\n\n"
        "Emit ONLY the fields that CHANGE from the previous plan (a partial plan). "
        "Unchanged fields are inherited; do not repeat them."
    )
    return LLMRequest(model=llm_model, system=SYSTEM, prompt=prompt, schema=_schema())


def interpret(
    provider: LLMProvider,
    model: SemanticModel,
    question: str,
    *,
    llm_model: str = "claude-sonnet-4-5",
) -> QueryIR:
    response = provider.complete(build_request(model, question, llm_model))
    return QueryIR.model_validate(response)


def interpret_followup(
    provider: LLMProvider,
    model: SemanticModel,
    previous: QueryIR,
    question: str,
    *,
    llm_model: str = "claude-sonnet-4-5",
) -> dict[str, Any]:
    """Return the raw PARTIAL plan (only changed fields) for merging — not a full IR."""
    return provider.complete(build_followup_request(model, previous, question, llm_model))
