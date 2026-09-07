"""Class 5 — degenerate execution, checked AFTER running.

Some failures are only visible in the result: a filter that matched nothing, an
all-null grouping key. These are annotated rather than presented as a confident zero,
so the caller never mistakes 'no rows matched' for a real answer of 0.
"""

from __future__ import annotations

from typing import Any

from .verdict import Caveat


def annotate_result(
    rows: list[dict[str, Any]], group_by: list[str], *, no_rows_matched: bool = False
) -> list[Caveat]:
    caveats: list[Caveat] = []
    # Two shapes of "nothing matched": a GROUPED query returns an empty result set; an
    # UNGROUPED aggregate returns ONE row of zeros/nulls. The old guard saw only the first and
    # presented the second as a confident 0. `no_rows_matched` (computed by the executor over
    # the same filters) catches the aggregate case so a zero is never mistaken for an answer.
    if not rows or no_rows_matched:
        caveats.append(
            Caveat(
                kind="degenerate",
                detail="no rows matched the filters — this is an empty result, not a zero",
            )
        )
        if not rows:
            return caveats
    # A projection rewrite (basket, frequency) changes the result SCHEMA: the IR's group_by
    # names columns of the PLAN, not of the answer — a basket result carries product_a/
    # product_b/pair_count, never the original grouping column. dict.get() returns None for an
    # ABSENT KEY exactly as for a null VALUE, so judging group_by directly would call every
    # basket grouping "all null" and discredit a correct answer. Judge only the columns the
    # ANSWER carries: require the key present before calling its values null (rows share one
    # schema).
    result_cols = set(rows[0])
    for col in group_by:
        if col not in result_cols:
            continue  # projected away by a rewrite; a different namespace than this result
        if all(r.get(col) is None for r in rows):
            caveats.append(
                Caveat(
                    kind="degenerate",
                    detail=f"every group has a null {col}; the grouping carries no signal",
                )
            )
    return caveats
