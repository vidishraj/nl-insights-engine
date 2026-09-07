"""Measure algebra — declarative measures defined over ROLES, not column names.

A measure names the roles it needs, its grain, and its additivity class. It resolves
to concrete SQL only when the required roles are bound *and verified*; otherwise it is
reported as unavailable with the missing roles named. Because measures are written over
roles, the identical definitions produce revenue on the enriched file (a verified
stored amount) and on raw UCI (the derived quantity × rate) with no change.
"""

from __future__ import annotations

from .model import MeasureBinding, ReturnsConvention
from .ontology import Role


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def returns_predicate(returns: ReturnsConvention) -> str | None:
    """A SQL predicate that is TRUE for return/cancellation rows, or None if unknown."""
    if returns.kind == "explicit_flag" and returns.column:
        return f"CAST({_q(returns.column)} AS INTEGER) = 1"
    if returns.kind == "explicit_category" and returns.column and returns.return_values:
        vals = ", ".join("'" + v.replace("'", "''") + "'" for v in returns.return_values)
        return f"{_q(returns.column)}::VARCHAR IN ({vals})"
    if returns.kind == "derived_key_prefix" and returns.column and returns.return_values:
        pfx = returns.return_values[0].replace("'", "''")
        return f"left({_q(returns.column)}::VARCHAR, {len(returns.return_values[0])}) = '{pfx}'"
    if returns.kind == "derived_negative_quantity" and returns.column:
        return f"{_q(returns.column)} < 0"
    return None


def build_measures(
    role_columns: dict[Role, list[str]],
    *,
    verified_amounts: set[str],
    returns: ReturnsConvention,
) -> list[MeasureBinding]:
    def first(role: Role) -> str | None:
        cols = role_columns.get(role, [])
        return cols[0] if cols else None

    quantity = first(Role.ADDITIVE_QUANTITY)
    rate = first(Role.MONETARY_RATE)
    amounts = role_columns.get(Role.MONETARY_AMOUNT, [])
    key = first(Role.TRANSACTION_KEY)

    measures: list[MeasureBinding] = []

    # --- revenue: prefer a column that ITSELF verified against quantity × rate; else the
    # derived quantity × rate. The verified column is selected EXPLICITLY (not cols[0]), so
    # the '[verified stored amount]' provenance is only ever claimed for the column that
    # actually passed the identity — never earned on one column and printed against another.
    verified_amount = next((a for a in amounts if a in verified_amounts), None)
    rev_note = "returns net out (their quantities are negative)"
    if verified_amount:
        rev_sql: str | None = f"sum({_q(verified_amount)})"
        rev_expr = f"sum({verified_amount})  [verified stored amount]"
        rev_missing: list[Role] = []
    elif quantity and rate:
        rev_sql = f"sum({_q(quantity)} * {_q(rate)})"
        rev_expr = "sum(additive_quantity × monetary_rate)  [derived]"
        rev_missing = []
    elif len(amounts) == 1:
        # (B) exactly ONE amount column and no quantity×rate to check it against: sum it AS
        # REPORTED rather than refuse. Refusing to sum the one money column on a file of bills
        # is the product failing while claiming rigour. The binder attaches a prose caveat
        # naming the column and saying the check could not be RUN (distinct from 'the check
        # failed'); with a SINGLE amount there is no competition for the binder's CLARIFY.
        col = amounts[0]
        rev_sql = f"sum({_q(col)})"
        rev_expr = f"sum({col})  [unverified amount, used as reported]"
        rev_missing = []
        rev_note = f"{col} summed as reported (could not be checked against quantity × rate)"
    else:
        # no verified amount, no quantity×rate, and either NO amount or SEVERAL competing ones.
        # Several competing amounts do not auto-pick one here — the binder CLARIFIES (C).
        rev_sql = None
        rev_expr = "sum(additive_quantity × monetary_rate) or sum(monetary_amount)"
        # coherence (item 4): report WHY revenue will not build rather than an empty missing list.
        rev_missing = [] if amounts else [Role.ADDITIVE_QUANTITY, Role.MONETARY_RATE]
        if len(amounts) > 1:
            rev_note = (
                f"several amount columns compete ({', '.join(amounts)}); which is revenue is "
                "ambiguous, so the binder asks rather than picking one"
            )

    # Net revenue nets returns automatically (negative-quantity rows subtract); gross
    # revenue excludes returns entirely.
    measures.append(
        MeasureBinding(
            name="net_revenue",
            expression=rev_expr,
            grain="row",
            additive=True,
            available=rev_sql is not None,
            sql=rev_sql,
            missing_roles=rev_missing,
            notes=[rev_note],
        )
    )
    ret_pred = returns_predicate(returns)
    gross_sql = (
        f"sum(CASE WHEN NOT ({ret_pred}) THEN {rev_sql[4:-1]} ELSE 0 END)"
        if rev_sql and ret_pred
        else rev_sql
    )
    measures.append(
        MeasureBinding(
            name="gross_revenue",
            expression=rev_expr + " over non-return rows",
            grain="row",
            additive=True,
            available=rev_sql is not None,
            sql=gross_sql,
            missing_roles=rev_missing,
            notes=[f"excludes returns ({returns.kind})"] if ret_pred else ["no returns convention"],
        )
    )

    # --- units sold ---
    measures.append(
        MeasureBinding(
            name="units_sold",
            expression="sum(additive_quantity)",
            grain="row",
            additive=True,
            available=quantity is not None,
            sql=f"sum({_q(quantity)})" if quantity else None,
            missing_roles=[] if quantity else [Role.ADDITIVE_QUANTITY],
        )
    )

    # --- order count: distinct transaction key ---
    measures.append(
        MeasureBinding(
            name="order_count",
            expression="count(distinct transaction_key)",
            grain="transaction",
            additive=False,
            available=key is not None,
            sql=f"count(DISTINCT {_q(key)})" if key else None,
            missing_roles=[] if key else [Role.TRANSACTION_KEY],
        )
    )

    # --- basket co-occurrence: needs a transaction key and a product entity ---
    product = next(
        (c for c in role_columns.get(Role.ENTITY_KEY, []) if c),
        None,
    )
    measures.append(
        MeasureBinding(
            name="basket_cooccurrence",
            expression="products appearing together within a transaction_key",
            grain="transaction",
            additive=False,
            available=key is not None and product is not None,
            sql=None,  # a self-join template the executor materialises per query
            missing_roles=([] if key and product else [Role.TRANSACTION_KEY, Role.ENTITY_KEY]),
            notes=["self-join line items on transaction_key; count distinct product pairs"],
        )
    )
    return measures
