"""Interpreter — a natural-language question to a typed Query IR (structured output).

Never free SQL, never raw data; the only component that asks the LLM about a QUESTION.
Follow-ups are handled as IR merge (diff plans), not by replaying chat history.
"""

from .interpret import (
    build_followup_request,
    build_request,
    interpret,
    interpret_followup,
    semantic_context,
)
from .ir import (
    Aggregation,
    CategoricalFilter,
    FrequencyConstraint,
    Grain,
    NumericFilter,
    PeriodComparison,
    QueryIR,
    TimeWindow,
    TopK,
)
from .merge import merge_ir

__all__ = [
    "Aggregation",
    "CategoricalFilter",
    "FrequencyConstraint",
    "Grain",
    "NumericFilter",
    "PeriodComparison",
    "QueryIR",
    "TimeWindow",
    "TopK",
    "build_followup_request",
    "build_request",
    "interpret",
    "interpret_followup",
    "merge_ir",
    "semantic_context",
]
