"""Executor — compile a bound plan to SQL, run on DuckDB, return an Answer.

The explanation is derived from the executed plan, so it cannot lie about what was
computed. Nothing here calls the LLM.
"""

from .answer import Answer
from .compile import compile_sql
from .execute import NeedsClarification, execute, summary_value

__all__ = ["Answer", "NeedsClarification", "compile_sql", "execute", "summary_value"]
