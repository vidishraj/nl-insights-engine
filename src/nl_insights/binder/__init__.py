"""Binder / verdict engine — the deterministic refusal spine.

No path from question to executed SQL bypasses ``bind``. It returns exactly one of
ANSWERABLE | REFUSE | CLARIFY | ANSWER_WITH_CAVEATS, every refusal carrying evidence.
"""

from .bind import bind
from .degenerate import annotate_result
from .verdict import BoundMeasure, BoundPlan, Caveat, Verdict, VerdictKind

__all__ = [
    "BoundMeasure",
    "BoundPlan",
    "Caveat",
    "Verdict",
    "VerdictKind",
    "annotate_result",
    "bind",
]
