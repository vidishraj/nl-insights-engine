"""Generate a SYNTHETIC retail dataset in the enriched SHAPE (not the client's data).

The client's enriched CSV is not publicly downloadable, so every ground-truth financial
test was skip-gated and could NEVER run for a grader (CRIT-14). This generates a ~500-row
file with the SAME column structure and conventions — product/non-product line types, an
explicit returns flag, a per-quarter completeness flag, anonymous (null-customer) rows, a
product that co-occurs across baskets, and one invoice that repeats a product on two lines
— so the executor golden tests run against committed, reproducible data and each fix's
test fails if the fix is removed.

    uv run python scripts/make_synthetic.py   # writes assets/nl-insights/synthetic-retail.csv

Deterministic (seeded), so the committed CSV and the tests' independently-computed ground
truth never drift.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "assets" / "nl-insights" / "synthetic-retail.csv"

HEADER = [
    "invoice_no",
    "invoice_type",
    "invoice_date",
    "invoice_year",
    "invoice_quarter",
    "invoice_month",
    "invoice_dow",
    "invoice_hour",
    "is_complete_quarter",
    "stock_code",
    "description",
    "line_type",
    "is_product_line",
    "is_revenue_line",
    "quantity",
    "unit_price",
    "line_revenue",
    "is_return",
    "customer_id",
    "is_identified_customer",
    "country",
    "is_country_known",
    "operator_note",
    "has_negative_price",
    "is_extreme_quantity",
]

# Real products. The last one, DELUXE, is deliberately high-value so that WITHOUT the
# product partition a POSTAGE line would out-rank real products (the partition trap).
PRODUCTS = [
    ("P001", "ALPHA MUG", 3.0),
    ("P002", "BETA NOTEBOOK", 5.0),
    ("P003", "GAMMA CANDLE", 2.5),
    ("P004", "DELTA LAMP", 9.0),
    ("P005", "EPSILON CLOCK", 12.0),
]
COUNTRIES = ["Wonderland", "Oz", "Narnia", "Atlantis"]

# Non-product line kinds. There must be >= 5 distinct line_type values (each with multiple
# rows) or the partition discovery's support threshold (fd_min_support=5) never fires and
# the line_type->is_product_line dependency is not learned — so 'top products by revenue'
# would silently rank POSTAGE. Each carries a small +/- amount so a bare total discloses it.
NON_PRODUCT = [
    ("POST", "DOTCOM POSTAGE", "POSTAGE", 5000.0),  # high, so it tops a naive ranking
    ("FEE", "BANK FEE", "FEE", -40.0),
    ("MAN", "MANUAL ADJUST", "MANUAL", -25.0),
    ("ADJ", "STOCK ADJUSTMENT", "ADJUSTMENT", -15.0),
    ("CAR", "CARRIAGE", "CARRIAGE", 30.0),
    ("DISC", "VOLUME DISCOUNT", "DISCOUNT", -12.0),
]

# Quarters: two COMPLETE (Q1, Q2) and one INCOMPLETE (Q3). The incomplete quarter carries
# real rows that MUST be excluded from growth — if the completeness clause is dropped, the
# growth numbers change (and the structural guard fires).
QUARTERS = [
    (
        "2021-Q1",
        1,
        [
            ("2021-01-11", "2021", "2021-01", 1, 10),
            ("2021-02-15", "2021", "2021-02", 0, 13),
            ("2021-03-08", "2021", "2021-03", 1, 12),
            ("2021-03-22", "2021", "2021-03", 1, 15),
        ],
    ),
    (
        "2021-Q2",
        1,
        [
            ("2021-04-12", "2021", "2021-04", 2, 9),
            ("2021-05-20", "2021", "2021-05", 3, 14),
            ("2021-05-27", "2021", "2021-05", 4, 16),
            ("2021-06-14", "2021", "2021-06", 0, 11),
        ],
    ),
    (
        "2021-Q3",
        0,
        [("2021-07-05", "2021", "2021-07", 0, 11), ("2021-08-09", "2021", "2021-08", 0, 10)],
    ),  # INCOMPLETE
]


def main() -> None:
    rng = random.Random(20260905)
    rows: list[dict[str, object]] = []
    inv = 0

    def add(
        inv_no,
        itype,
        date,
        yr,
        mon,
        q,
        complete,
        dow,
        hour,
        code,
        desc,
        ltype,
        is_prod,
        qty,
        price,
        is_ret,
        cust,
        country,
    ):
        rows.append(
            {
                "invoice_no": inv_no,
                "invoice_type": itype,
                "invoice_date": f"{date} {hour:02d}:00:00",
                "invoice_year": yr,
                "invoice_quarter": q,
                "invoice_month": mon,
                "invoice_dow": dow,
                "invoice_hour": hour,
                "is_complete_quarter": complete,
                "stock_code": code,
                "description": desc,
                "line_type": ltype,
                "is_product_line": is_prod,
                "is_revenue_line": 1,
                "quantity": qty,
                "unit_price": price,
                "line_revenue": round(qty * price, 2),
                "is_return": is_ret,
                "customer_id": cust or "",
                "is_identified_customer": 1 if cust else 0,
                "country": country,
                "is_country_known": 1,
                "operator_note": f"note {inv_no}",
                "has_negative_price": 1 if price < 0 else 0,
                "is_extreme_quantity": 1 if abs(qty) > 100 else 0,
            }
        )

    # Per country we ramp PRODUCT revenue up from Q1 to Q2 by a country-specific factor,
    # so growth has a clear, independently-computable ranking that reflects product sales.
    growth_factor = {"Wonderland": 1.1, "Oz": 3.0, "Narnia": 1.5, "Atlantis": 2.2}
    combo = 0  # (quarter, country) index; drives the deterministic duplicate-line case
    for qlabel, complete, dates in QUARTERS:
        for country in COUNTRIES:
            combo += 1
            base = (
                4
                if qlabel == "2021-Q1"
                else (int(4 * growth_factor[country]) if qlabel == "2021-Q2" else 5)
            )
            for date, yr, mon, dow, hour in dates:
                for pi, (code, desc, price) in enumerate(PRODUCTS):
                    inv += 1
                    inv_no = f"INV{inv:05d}"
                    anon = inv % 7 in (0, 1)  # ~28% anonymous, deterministic
                    cust = None if anon else f"CUST{(inv % 37):03d}"
                    qty = base + pi
                    add(
                        inv_no,
                        "invoice",
                        date,
                        yr,
                        mon,
                        qlabel,
                        complete,
                        dow,
                        hour,
                        code,
                        desc,
                        "PRODUCT",
                        1,
                        qty,
                        price,
                        0,
                        cust,
                        country,
                    )
                    # basket: pair the first two products in the same invoice so they
                    # co-occur; and for one combo in three, repeat P001 on a 2nd line so
                    # line-pairs (count(*)) exceed distinct baskets (count(DISTINCT txn)).
                    if pi == 0:
                        add(
                            inv_no,
                            "invoice",
                            date,
                            yr,
                            mon,
                            qlabel,
                            complete,
                            dow,
                            hour,
                            PRODUCTS[1][0],
                            PRODUCTS[1][1],
                            "PRODUCT",
                            1,
                            base + 1,
                            PRODUCTS[1][2],
                            0,
                            cust,
                            country,
                        )
                        if combo % 3 == 0:  # deterministic duplicate P001 line
                            add(
                                inv_no,
                                "invoice",
                                date,
                                yr,
                                mon,
                                qlabel,
                                complete,
                                dow,
                                hour,
                                code,
                                desc,
                                "PRODUCT",
                                1,
                                1,
                                price,
                                0,
                                cust,
                                country,
                            )
            # non-product lines ONCE per (quarter, country): >= 6 distinct line_type values
            # (so the partition is discovered) and money a bare total must disclose. Placed
            # once per quarter so they do not swamp the product growth signal, but their
            # total (grouped by description) still tops a naive product ranking.
            d0, yr0, mon0, dow0, hour0 = dates[0]
            for ncode, ndesc, ntype, namt in NON_PRODUCT:
                inv += 1
                add(
                    f"INV{inv:05d}",
                    "invoice",
                    d0,
                    yr0,
                    mon0,
                    qlabel,
                    complete,
                    dow0,
                    hour0,
                    ncode,
                    ndesc,
                    ntype,
                    0,
                    1,
                    namt,
                    0,
                    None,
                    country,
                )

    # A tiny-base country present in both complete quarters, so a growth ranking has a group
    # whose base is a negligible share of the period total: a large % over a small base is the
    # trap the base disclosure must quantify (MAJ-alpha). One product line each quarter.
    for qlabel, complete, dates in QUARTERS[:2]:  # Q1, Q2 (both complete)
        d0, yr0, mon0, dow0, hour0 = dates[0]
        inv += 1
        qty = 1 if qlabel == "2021-Q1" else 4  # grows 1 -> 4 off a tiny base
        add(
            f"INV{inv:05d}",
            "invoice",
            d0,
            yr0,
            mon0,
            qlabel,
            complete,
            dow0,
            hour0,
            "P001",
            "ALPHA MUG",
            "PRODUCT",
            1,
            qty,
            3.0,
            0,
            "CUST001",
            "Sark",
        )

    # A country that RECOVERS from a net-negative base: Q1 is negative (a large return
    # outweighs a small sale), Q2 is positive. A percentage change over a negative base
    # inverts its sign, so growth must NULL it (never rank it as a decline) and disclose the
    # negative base as such — the Bahrain case, made reproducible in the committed data.
    d1 = QUARTERS[0][2][0]  # a Q1 date
    d2 = QUARTERS[1][2][0]  # a Q2 date
    inv += 1
    add(
        f"INV{inv:05d}",
        "invoice",
        d1[0],
        d1[1],
        d1[2],
        "2021-Q1",
        1,
        d1[3],
        d1[4],
        "P001",
        "ALPHA MUG",
        "PRODUCT",
        1,
        2,
        3.0,
        0,
        "CUST002",
        "Tortuga",
    )
    inv += 1  # a big return the same quarter -> net Q1 negative
    add(
        f"INV{inv:05d}",
        "cancellation",
        d1[0],
        d1[1],
        d1[2],
        "2021-Q1",
        1,
        d1[3],
        d1[4],
        "P001",
        "ALPHA MUG",
        "PRODUCT",
        1,
        -100,
        3.0,
        1,
        "CUST002",
        "Tortuga",
    )
    inv += 1
    add(
        f"INV{inv:05d}",
        "invoice",
        d2[0],
        d2[1],
        d2[2],
        "2021-Q2",
        1,
        d2[3],
        d2[4],
        "P001",
        "ALPHA MUG",
        "PRODUCT",
        1,
        50,
        3.0,
        0,
        "CUST002",
        "Tortuga",
    )

    # a handful of RETURNS (is_return=1, negative), which basket/co-occurrence must exclude.
    for i in range(6):
        inv += 1
        code, desc, price = PRODUCTS[i % len(PRODUCTS)]
        add(
            f"INV{inv:05d}",
            "cancellation",
            "2021-04-18",
            "2021",
            "2021-04",
            "2021-Q2",
            1,
            0,
            10,
            code,
            desc,
            "PRODUCT",
            1,
            -(2 + i),
            price,
            1,
            f"CUST{(i % 37):03d}",
            COUNTRIES[i % len(COUNTRIES)],
        )

    rng.shuffle(rows)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
