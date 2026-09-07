"""Evidence-based date-order resolution.

The ambiguity that quietly corrupts analytics is ``01/02/2019``: is it 1 Feb or 2 Jan?
We never assume a locale. We look at the actual values: if any first component exceeds
12 the order is day-first; if any second component does, it is month-first. If nothing
in the column disambiguates, we say so (``AMBIGUOUS``) rather than pick silently — the
semantic layer can then surface a caveat instead of trusting a coin-flip.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

# Two/one-digit, sep, two/one-digit, sep, 1-4 digit  (optionally followed by a time).
_TRIPLE = re.compile(r"^\s*(\d{1,4})([/.\-])(\d{1,2})\2(\d{1,4})(?:[ T].*)?$")


class DateOrder(StrEnum):
    ISO = "iso"  # year-first, e.g. 2019-02-01 — unambiguous
    DAY_FIRST = "day_first"  # dd/mm/yyyy
    MONTH_FIRST = "month_first"  # mm/dd/yyyy
    AMBIGUOUS = "ambiguous"  # looks like a date but nothing settles day vs month
    NOT_A_DATE = "not_a_date"


@dataclass(frozen=True)
class DateResolution:
    order: DateOrder
    evidence: str
    considered: int


def resolve_date_order(values: Iterable[str]) -> DateResolution:
    samples = [v.strip() for v in values if v and v.strip()]
    if not samples:
        return DateResolution(DateOrder.NOT_A_DATE, "no non-empty values", 0)

    max_a = max_b = 0
    ev_day = ev_month = ""
    matched = 0
    iso_like = 0

    for value in samples:
        m = _TRIPLE.match(value)
        if not m:
            continue
        matched += 1
        a, b = int(m.group(1)), int(m.group(3))
        # Year-first (first field is a 4-digit year) is ISO order, unambiguous.
        if len(m.group(1)) == 4 or a > 31:
            iso_like += 1
            continue
        if a > max_a:
            max_a, ev_day = a, value
        if b > max_b:
            max_b, ev_month = b, value

    if matched == 0:
        return DateResolution(DateOrder.NOT_A_DATE, "no value matched a date pattern", len(samples))
    if iso_like == matched:
        return DateResolution(DateOrder.ISO, "all values are year-first (ISO)", matched)

    # A component > 12 can only be a day, which fixes the order.
    if max_a > 12 and max_b <= 12:
        return DateResolution(
            DateOrder.DAY_FIRST, f"{ev_day!r}: first component {max_a} > 12", matched
        )
    if max_b > 12 and max_a <= 12:
        return DateResolution(
            DateOrder.MONTH_FIRST, f"{ev_month!r}: second component {max_b} > 12", matched
        )
    if max_a > 12 and max_b > 12:
        return DateResolution(
            DateOrder.AMBIGUOUS, "both components exceed 12 — not a consistent date column", matched
        )
    return DateResolution(
        DateOrder.AMBIGUOUS, "no component exceeds 12; day vs month cannot be decided", matched
    )
