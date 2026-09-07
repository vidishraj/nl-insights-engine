"""Run a BoundPlan on DuckDB and assemble the Answer with a plan-derived explanation."""

from __future__ import annotations

from typing import Any

import duckdb

from ..binder.degenerate import annotate_result
from ..binder.verdict import BoundPlan, Caveat
from ..interpreter.ir import CategoricalFilter, NumericFilter
from ..semantic.model import ColumnBinding, SemanticModel
from ..semantic.ontology import Role
from .answer import Answer
from .compile import (
    _categorical,
    _lit,
    _numeric,
    _period_expr,
    _q,
    _time,
    compile_sql,
    complete_period_clause,
    last_n_interval,
)


class WindowClaimError(RuntimeError):
    """Raised when a time-window caveat was emitted but the compiler produced no matching
    time clause — a stated invariant (we never claim a filter the SQL does not contain).
    An unconditional raise, not an assert, so ``-O`` cannot strip the guarantee."""


class NeedsClarification(Exception):
    """A DATA-dependent clarification the binder could not make without the DB: a growth
    comparison names a period that is not in the data, or the data has fewer than two
    periods. The binder is still the structural refusal spine; this covers the one case
    whose evidence only exists at execution time. The pipeline turns it into a CLARIFY
    verdict, so the answer is never a silent zero/NULL over a period that isn't there."""

    def __init__(
        self,
        question: str,
        options: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(question)
        self.question = question
        self.options = options or []
        self.evidence = evidence or {}


_FREQ_OPS = {"eq": "=", "lte": "<=", "gte": ">="}
_GROWTH_KIND = {  # per PeriodComparison.kind: how the ranked value is derived from a,b
    "growth": "(b - a) / a",  # relative growth (b-a)/a
    "change": "(b - a)",  # absolute delta
    "ratio": "(b / a)",  # ratio b/a
}
# Measures whose value depends on whether non-product rows (postage, fees, adjustments)
# are in scope. When one of these is computed WITHOUT the fact partition, the executor
# quantifies exactly what non-product money the answer contains (see _disclose_...).
_REVENUE_MEASURES = frozenset({"net_revenue", "gross_revenue"})


def execute(
    con: duckdb.DuckDBPyConnection,
    plan: BoundPlan,
    model: SemanticModel,
    caveats: list[Caveat],
) -> Answer:
    event_time = model.first_in_role(Role.EVENT_TIME)
    et_name = event_time.column if event_time else None  # the column, for human text
    et = _event_time_sql(event_time, model)  # a SQL expression (quoted, or a CAST)

    period_comparison = bool(plan.ir.period_comparison and et)
    period_labels: tuple[str, str] | None = None
    if plan.ir.basket:
        sql = _compile_basket(plan, model, et)
    elif plan.ir.frequency:
        sql = _compile_frequency(plan, et)
    elif period_comparison:
        assert et is not None
        pa, pb, _avail = _resolve_periods(con, plan, et)  # may raise NeedsClarification
        period_labels = (pa, pb)
        sql = _period_comparison_sql(plan, et, pa, pb)
    else:
        sql = compile_sql(plan, et)

    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, r, strict=False)) for r in cur.fetchall()]

    # share-of-total: each group's fraction of the total over ALL groups. The denominator
    # is computed in SQL over the UN-LIMITED grouped result (see _share_denominator), never
    # from the post-LIMIT `rows` — summing the fetched top-k made every share sum to 1.0 by
    # construction (a 'top 5' each reported ~5x its true share).
    if plan.ir.share_of_total and plan.measures and not plan.ir.frequency and not period_comparison:
        m = plan.measures[0].name
        total = _share_denominator(con, plan, et)
        for row in rows:
            row["share"] = (row.get(m) or 0) / total if total else None
        if "share" not in columns:
            columns.append("share")

    # frequency: the matching group's measure as a SHARE of the total, stated explicitly
    # (e.g. "customers who bought once account for 4.5% of revenue").
    if plan.ir.frequency and rows:
        row = rows[0]
        grp, tot = row.get("group_measure") or 0, row.get("total_measure") or 0
        row["share"] = (grp / tot) if tot else None
        if "share" not in columns:
            columns.append("share")

    # Detect the zero-match AGGREGATE: an ungrouped total over filters that matched nothing
    # returns one row of zeros, which must not read as a confident 0. A cheap EXISTS over the
    # SAME filters tells us whether any row matched. (Grouped queries surface this as an empty
    # result set already; basket/frequency are different shapes.)
    # A pure COUNT aggregation is exempt: 'how many X in 1999' = 0 is the TRUE count when
    # nothing matched, not a false zero. Only sum/avg (and revenue) turn an empty match into an
    # ambiguous zero that must be caveated.
    count_only = (
        bool(plan.ir.aggregations)
        and all(a.func == "count" for a in plan.ir.aggregations)
        and not plan.ir.measures
        and not plan.ir.distinct_count_of
    )
    no_rows_matched = False
    if (
        not plan.group_by
        and not plan.ir.basket
        and not plan.ir.frequency
        and not period_comparison
        and not count_only
    ):
        where = _where(plan, et)
        if where:  # only meaningful when the query actually filters
            hit = con.execute(
                f"SELECT EXISTS(SELECT 1 FROM {_q(plan.table)} WHERE {where})"
            ).fetchone()
            no_rows_matched = hit is not None and not hit[0]
    degen = annotate_result(rows, plan.group_by, no_rows_matched=no_rows_matched)

    # Structural guard against the exact bug this path once had: a "window" caveat must never
    # claim a filter the SQL does not contain. If the binder announced a time window, the
    # compiler must have produced at least one time clause for it — otherwise we would be
    # reporting on work that was never done. This is a stated correctness guarantee, so it is
    # an unconditional raise, NOT an assert (asserts are stripped under -O / PYTHONOPTIMIZE,
    # which would let the exact bug back in with no signal). Failing loud beats a wrong answer.
    if (
        plan.ir.time
        and et
        and any(c.kind == "window" for c in caveats)
        and not _time(plan.ir.time, et, plan.table)
    ):
        raise WindowClaimError("a time-window caveat was emitted but no time clause was compiled")

    # The SAME guarantee for the incomplete-period exclusion (period comparison): if the
    # binder disclosed 'incomplete periods excluded via X' (a 'period' caveat), the compiled
    # SQL MUST carry that flag's predicate. Dropping the clause would silently include partial
    # periods while still claiming they were excluded — a false disclosure in the subsystem
    # whose entire pitch is disclosure discipline. Unconditional raise (-O safe), like the
    # window guard above; a mutant that removes the clause fails loud instead of shipping a lie.
    if (
        plan.require_complete_period_flag
        and any(c.kind == "period" for c in caveats)
        and complete_period_clause(plan.require_complete_period_flag) not in sql
    ):
        raise WindowClaimError(
            "an incomplete-period caveat was emitted but no complete-period clause was compiled"
        )

    # Disclose the ROLLING window's actual anchor and whether it covers the whole span. The
    # window is anchored on the data's latest date (see compile._time), which only an honest
    # answer states — and it turns a coincidentally-right total into an explained one.
    window_disclosure: Caveat | None = None
    if plan.ir.time and plan.ir.time.last_n and et:
        window_disclosure = _disclose_last_n_window(con, plan, et)

    # Disclose (and QUANTIFY) the scope assumption: a revenue total computed without the
    # fact partition still contains postage/fees/adjustments. Only the standard path can
    # carry a revenue measure; basket/frequency are different question shapes.
    disclosure: Caveat | None = None
    if not plan.ir.basket and not plan.ir.frequency and not period_comparison:
        disclosure = _disclose_non_fact_rows(con, plan, model, et)

    measure_expr = {mb.name: mb.expression for mb in model.measures}
    assumptions = [c.detail for c in caveats if c.kind == "assumption"]
    other_caveats = [c.detail for c in caveats if c.kind != "assumption"] + [
        c.detail for c in degen
    ]
    # Route the non-fact disclosure by its OWN kind, exactly like every binder caveat above. It is
    # now a 'partition' caveat (a limitation of the number, not an interpretation), so it belongs
    # with the caveats rather than the assumptions where it used to be hardcoded.
    if disclosure is not None:
        (assumptions if disclosure.kind == "assumption" else other_caveats).append(
            disclosure.detail
        )
    if window_disclosure is not None:
        other_caveats.append(window_disclosure.detail)
    # Basket: state what pair_count means and what is in scope, so the number is not read as
    # a raw line-pair count over every line kind (it counts DISTINCT transactions, product
    # lines only, returns excluded).
    if plan.ir.basket:
        other_caveats.append(
            "pair_count is the number of distinct transactions containing both products; "
            "restricted to product lines, returns excluded"
        )
    # Growth: name the two periods actually compared and the ranking, so a defaulted period
    # pair is never silent (the SQL ranks by exactly this).
    if period_comparison and period_labels is not None:
        pa, pb = period_labels
        kind = plan.ir.period_comparison.kind  # type: ignore[union-attr]
        other_caveats.append(
            f"{kind} of {plan.measures[0].name if plan.measures else 'the measure'} "
            f"from {pa} to {pb}; groups present in both periods, ranked by {kind}"
        )
        # MAJ-alpha (+ negative-base): a huge % off a tiny base is arithmetically right but
        # misleading, and a % over a NEGATIVE base inverts its sign. Disclose both, with
        # figures, over every ranked-eligible group — the user decides, we never silently drop
        # or mislabel. Same channel and quantified style as the non-fact disclosure.
        base_caveat = _disclose_growth_base_anomalies(con, plan, et, pa, pb)
        if base_caveat is not None:
            assumptions.append(base_caveat.detail)
        # MAJ-beta: if no period-completeness flag was discovered, we could not exclude a
        # partial period. Silence would read as "no incomplete periods"; say instead that we
        # had no way to tell, so a partial latest period may depress the final period.
        if not plan.require_complete_period_flag:
            other_caveats.append(
                "no period-completeness flag was found for this dataset, so an incomplete "
                f"final period is NOT excluded; if {pb} is still in progress its values are "
                "understated and the ranking may be affected"
            )

    used = set(plan.group_by) | (
        {plan.ir.distinct_count_of} if plan.ir.distinct_count_of else set()
    )
    coverage = {c: model.coverage[c] for c in used if c in model.coverage}

    return Answer(
        rows=rows,
        columns=columns,
        sql=sql,
        explanation=_explain(plan, et_name),
        formula=[f"{m.name} = {measure_expr.get(m.name, m.sql)}" for m in plan.measures],
        filters_applied=_describe_filters(plan, et_name),
        grain=_answer_grain(plan),
        assumptions=assumptions,
        caveats=other_caveats,
        coverage=coverage,
        plan=plan,
    )


def _event_time_sql(binding: ColumnBinding | None, model: SemanticModel) -> str | None:
    """The SQL expression for the event_time: a quoted column, or a TRY_CAST when the
    column is text-but-parseable. Returns None when no event_time is bound (time queries
    have already been refused by the binder)."""
    if binding is None:
        return None
    return model.temporal_cast_sql.get(binding.column) or _q(binding.column)


def _compile_frequency(plan: BoundPlan, event_time: str | None) -> str:
    freq = plan.ir.frequency
    assert freq is not None
    mexpr = plan.measures[0].sql if plan.measures else "count(*)"
    # Exclude the anonymous pseudo-entity: SQL GROUP BY collapses every NULL-entity row into
    # ONE group whose measure would otherwise land in `total_measure` (the denominator). When
    # a large fraction of rows carry no entity id, the share came out materially low AND the
    # coverage caveat named a scope ('the N% of rows that carry the entity') the SQL never
    # applied. This predicate makes the denominator match the caveat.
    parts = [f"{_q(freq.entity)} IS NOT NULL"]
    inner = _where(plan, event_time)
    if inner:
        parts.append(inner)
    where_sql = " WHERE " + " AND ".join(parts)
    per = (
        f"SELECT {_q(freq.entity)} AS e, count(DISTINCT {_q(freq.count_of)}) AS purchases, "
        f"{mexpr} AS m FROM {_q(plan.table)}{where_sql} GROUP BY {_q(freq.entity)}"
    )
    op = _FREQ_OPS[freq.op]  # direct index: an unmapped op raises, never silently defaults
    return (
        f"WITH per AS ({per}) "
        f"SELECT count(*) AS entity_count, sum(m) AS group_measure, "
        f"(SELECT sum(m) FROM per) AS total_measure "
        f"FROM per WHERE purchases {op} {freq.n}"
    )


def _basket_scope_predicates(plan: BoundPlan, model: SemanticModel) -> list[str]:
    """The filters that make a basket a basket of real purchases: restrict to product
    lines (postage/fees/adjustments are not products bought together) and exclude returns
    (a return is not a 'bought together' event). Returned so the executor can both apply
    them in SQL and DISCLOSE them, so the two never disagree."""
    preds: list[str] = []
    for pd in model.partition_dimensions:
        if pd.fact_value:
            preds.append(f"{_q(pd.column)}::VARCHAR IN ({_lit(pd.fact_value)})")
    ret = model.returns
    if ret.kind not in {"none", ""} and ret.column and ret.return_values:
        vals = ", ".join(_lit(v) for v in ret.return_values)
        preds.append(f"{_q(ret.column)}::VARCHAR NOT IN ({vals})")
    return preds


def _compile_basket(plan: BoundPlan, model: SemanticModel, event_time: str | None) -> str:
    """Products co-occurring within a transaction: a self-join on the transaction key.

    ``count(DISTINCT a.txn)`` counts BASKETS (distinct transactions containing both), not
    line-pairs — a product appearing on two lines of one invoice must not double-count.
    The lines CTE is restricted to product lines with returns excluded (see
    _basket_scope_predicates) so co-occurrence is over real purchases only.
    """
    from ..semantic.ontology import EntityKind

    key = model.first_in_role(Role.TRANSACTION_KEY)
    product = next(
        (
            b
            for b in model.bindings
            if b.role == Role.ENTITY_KEY and b.entity == EntityKind.PRODUCT and not b.refuted
        ),
        None,
    )
    if key is None or product is None:  # pragma: no cover - binder refuses this first
        return "SELECT NULL WHERE FALSE"
    k, p = _q(key.column), _q(product.column)
    parts = _basket_scope_predicates(plan, model)
    inner = _where(plan, event_time)
    if inner:
        parts.append(inner)
    where_sql = (" WHERE " + " AND ".join(parts)) if parts else ""
    base = f"SELECT DISTINCT {k} AS txn, {p} AS prod FROM {_q(plan.table)}{where_sql}"
    limit = plan.ir.top_k.k if plan.ir.top_k else 20
    return (
        f"WITH lines AS ({base}) "
        f"SELECT a.prod AS product_a, b.prod AS product_b, count(DISTINCT a.txn) AS pair_count "
        f"FROM lines a JOIN lines b ON a.txn = b.txn AND a.prod < b.prod "
        f"GROUP BY a.prod, b.prod ORDER BY pair_count DESC LIMIT {limit}"
    )


def _share_denominator(
    con: duckdb.DuckDBPyConnection, plan: BoundPlan, event_time: str | None
) -> float:
    """The share-of-total denominator: the sum of the primary measure over ALL groups
    (the same grouped query WITHOUT the top-k LIMIT), computed in SQL. Summing the fetched
    post-LIMIT rows instead made every share sum to 1.0 by construction."""
    m = plan.measures[0].name
    ir2 = plan.ir.model_copy(update={"top_k": None, "share_of_total": False})
    plan2 = plan.model_copy(update={"ir": ir2})
    inner = compile_sql(plan2, event_time)
    row = con.execute(f"SELECT sum({_q(m)}) FROM ({inner}) AS _all").fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def _resolve_periods(
    con: duckdb.DuckDBPyConnection, plan: BoundPlan, event_time: str
) -> tuple[str, str, list[str]]:
    """Resolve the two period labels a growth comparison ranks over, against the DATA.

    Returns (period_a, period_b, available_labels). Defaults are the two most-recent
    complete periods. An explicitly-named period that is not in the data — or a dataset
    with fewer than two periods — raises NeedsClarification, which the pipeline turns into
    a CLARIFY: a 'grew most' question must never silently rank over a period that isn't
    there. Available labels honour the query's own filters (incl. the incomplete-period
    exclusion the binder set)."""
    pc = plan.ir.period_comparison
    assert pc is not None
    pexpr = _period_expr(event_time, pc.grain)
    where = _where(plan, event_time)
    parts = [where] if where else []
    parts.append(f"{pexpr} IS NOT NULL")
    sel_where = " WHERE " + " AND ".join(parts)
    avail = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT {pexpr} AS q FROM {_q(plan.table)}{sel_where} ORDER BY q"
        ).fetchall()
    ]
    pa, pb = pc.period_a, pc.period_b
    named = [x for x in (pa, pb) if x is not None]
    missing = [x for x in named if x not in avail]
    if missing:
        raise NeedsClarification(
            f"{', '.join(missing)} is not in the data. "
            f"Available {pc.grain}s are {', '.join(avail) or '(none)'} — which two do you mean?",
            options=avail,
            evidence={"requested": named, "available_periods": avail, "grain": pc.grain},
        )
    if pa is None and pb is None:
        if len(avail) < 2:
            raise NeedsClarification(
                f"A growth comparison needs two {pc.grain}s, but the data has "
                f"{len(avail)} ({', '.join(avail) or 'none'}).",
                options=avail,
                evidence={"available_periods": avail, "grain": pc.grain},
            )
        return avail[-2], avail[-1], avail
    if pb is None:  # only the earlier period named → later = latest distinct one
        later = [x for x in avail if x != pa]
        if not later:
            raise NeedsClarification(
                f"Only one {pc.grain} ({pa}) is available; need a second to compare.",
                options=avail,
                evidence={"available_periods": avail, "grain": pc.grain},
            )
        return pa, later[-1], avail  # type: ignore[return-value]
    if pa is None:  # only the later period named → earlier = most recent distinct before it
        earlier = [x for x in avail if x != pb]
        if not earlier:
            raise NeedsClarification(
                f"Only one {pc.grain} ({pb}) is available; need a second to compare.",
                options=avail,
                evidence={"available_periods": avail, "grain": pc.grain},
            )
        return earlier[-1], pb, avail
    return pa, pb, avail


def _period_comparison_sql(plan: BoundPlan, event_time: str, period_a: str, period_b: str) -> str:
    """Pivot the two resolved periods per group and rank by the growth/change/ratio the IR
    asked for. Only groups present in BOTH periods get a defined value; the rest sort last
    (NULLS LAST). This is where growth is actually COMPUTED — the old path returned the
    per-period rows unranked and left the arithmetic to the user."""
    pc = plan.ir.period_comparison
    assert pc is not None and plan.measures
    pexpr = _period_expr(event_time, pc.grain)
    mexpr = plan.measures[0].sql
    gcols = [_q(g) for g in plan.group_by]
    gsel = (", ".join(gcols) + ", ") if gcols else ""
    where = _where(plan, event_time)
    per_parts = [where] if where else []
    per_parts.append(f"{pexpr} IN ({_lit(period_a)}, {_lit(period_b)})")
    per_where = " WHERE " + " AND ".join(per_parts)
    per = (
        f"SELECT {gsel}{pexpr} AS period, {mexpr} AS m "
        f"FROM {_q(plan.table)}{per_where} GROUP BY {gsel}{pexpr}"
    )
    piv_group = f" GROUP BY {', '.join(gcols)}" if gcols else ""
    piv = (
        f"SELECT {gsel}"
        f"MAX(CASE WHEN period = {_lit(period_a)} THEN m END) AS period_a_value, "
        f"MAX(CASE WHEN period = {_lit(period_b)} THEN m END) AS period_b_value "
        f"FROM per{piv_group}"
    )
    if pc.kind == "change":
        growth = "(period_b_value - period_a_value)"  # absolute delta is fine over any base
    elif pc.kind == "ratio":
        # base <= 0, not just = 0: a ratio/percentage over a NEGATIVE base inverts its sign
        # (a recovery from -200 to +500 would read as a large decline). It is undefined, not
        # a real change — NULL it and disclose the negative base, never rank it.
        growth = (
            "CASE WHEN period_a_value IS NULL OR period_a_value <= 0 THEN NULL "
            "ELSE period_b_value / period_a_value END"
        )
    else:  # growth
        growth = (
            "CASE WHEN period_a_value IS NULL OR period_a_value <= 0 THEN NULL "
            "ELSE (period_b_value - period_a_value) / period_a_value END"
        )
    direction = "ASC" if (plan.ir.top_k and plan.ir.top_k.direction == "asc") else "DESC"
    k = plan.ir.top_k.k if plan.ir.top_k else 10
    return (
        f"WITH per AS ({per}), piv AS ({piv}) "
        f"SELECT {gsel}period_a_value, period_b_value, {growth} AS growth "
        f"FROM piv ORDER BY growth {direction} NULLS LAST LIMIT {k}"
    )


def _answer_grain(plan: BoundPlan) -> str:
    if plan.ir.period_comparison:
        return str(plan.ir.period_comparison.grain)
    t = plan.ir.time
    return str(t.grain) if t and t.grain else "row"


def _disclose_growth_base_anomalies(
    con: duckdb.DuckDBPyConnection,
    plan: BoundPlan,
    event_time: str | None,
    period_a: str,
    period_b: str,
) -> Caveat | None:
    """Disclose the two ways a growth ranking's BASE misleads, computed over every group
    present in both periods (not only the shipped top-k, since a negative base is NULL-growth
    and drops out of it):

    - a NEGATIVE base: a percentage change over it inverts its sign (a recovery from -200 to
      +500 would read as a large decline), so it is NULL-growth and NOT ranked — state that,
      with the figure, rather than let its absence be silent.
    - a small POSITIVE base (a negligible share of the base-period total): still ranked, but a
      huge percentage over it is volatile.

    Speaks even when the base-period total is <= 0 (every base anomalous) — the disclosure
    exists to prevent exactly that confusion, so it must not go silent there."""
    pc = plan.ir.period_comparison
    if not plan.measures or pc is None or event_time is None:
        return None
    pexpr = _period_expr(event_time, pc.grain)
    mexpr = plan.measures[0].sql
    gcols = [_q(g) for g in plan.group_by]
    gsel = (", ".join(gcols) + ", ") if gcols else ""
    where = _where(plan, event_time)
    per_parts = [where] if where else []
    per_parts.append(f"{pexpr} IN ({_lit(period_a)}, {_lit(period_b)})")
    per = (
        f"SELECT {gsel}{pexpr} AS period, {mexpr} AS m "
        f"FROM {_q(plan.table)} WHERE {' AND '.join(per_parts)} GROUP BY {gsel}{pexpr}"
    )
    piv_group = f" GROUP BY {', '.join(gcols)}" if gcols else ""
    # group label + base value, over groups present in BOTH periods (the ranked-eligible set)
    label_expr = f"concat_ws('/', {', '.join(gcols)})" if gcols else "'group'"
    rows = con.execute(
        f"WITH per AS ({per}), piv AS ("
        f"SELECT {gsel}MAX(CASE WHEN period = {_lit(period_a)} THEN m END) AS a, "
        f"MAX(CASE WHEN period = {_lit(period_b)} THEN m END) AS b FROM per{piv_group}) "
        f"SELECT {label_expr} AS label, a FROM piv WHERE a IS NOT NULL AND b IS NOT NULL"
    ).fetchall()
    total_a = sum(float(a) for _, a in rows if a is not None)
    threshold = 0.01  # a positive base under 1% of the base-period total is "small"
    negative: list[tuple[str, float]] = []
    small: list[tuple[str, float, float]] = []
    for label, a in rows:
        if a is None:
            continue
        a = float(a)
        if a <= 0:
            negative.append((str(label), a))
        elif total_a > 0 and a / total_a < threshold:
            small.append((str(label), a, a / total_a))
    if not negative and not small:
        return None

    def _listed(items: list[tuple[str, float]]) -> str:
        head = ", ".join(f"{lab} base {base:,.2f}" for lab, base in items[:3])
        more = len(items) - min(3, len(items))
        return head + (f", +{more} more" if more > 0 else "")

    segments: list[str] = []
    if negative:
        segments.append(
            f"{len(negative)} group(s) have a NEGATIVE {period_a} base ({_listed(negative)}); a "
            "percentage change over a negative base is undefined, so they are NOT ranked"
        )
    if small:
        listed = ", ".join(
            f"{lab} base {base:,.2f} ({frac:.3%} of {period_a})" for lab, base, frac in small[:3]
        )
        more = len(small) - min(3, len(small))
        tail = f", +{more} more" if more > 0 else ""
        segments.append(
            f"{len(small)} ranked group(s) have a {period_a} base under {threshold:.0%} of that "
            f"period's total ({listed}{tail}); a large percentage over so small a base is volatile"
        )
    return Caveat(
        kind="assumption",
        detail="; ".join(segments),
        metrics={
            "negative_base_groups": float(len(negative)),
            "small_base_groups": float(len(small)),
        },
    )


def _disclose_last_n_window(
    con: duckdb.DuckDBPyConnection, plan: BoundPlan, event_time: str
) -> Caveat | None:
    """State the ROLLING window a `last N <grain>` query actually applied: its anchor is the
    LATEST date in the data (not today), and if the window reaches past the earliest date the
    answer is the whole dataset. Without this, a data-relative anchor is an unstated
    assumption and a whole-span 'filter' looks like a real restriction."""
    win = plan.ir.time
    interval = last_n_interval(win) if win else None
    if win is None or interval is None:
        return None
    tbl = _q(plan.table)
    row = con.execute(
        f"SELECT CAST(min({event_time}) AS DATE), CAST(max({event_time}) AS DATE), "
        f"CAST((SELECT max({event_time}) FROM {tbl}) - {interval} AS DATE) FROM {tbl}"
    ).fetchone()
    if row is None or row[1] is None:
        return None
    lo, hi, window_start = row
    unit = f"{win.last_n} {win.grain}" + ("s" if (win.last_n or 0) != 1 else "")
    detail = f"last {unit} relative to the latest date in the data ({hi})"
    if window_start is not None and lo is not None and window_start <= lo:
        detail += (
            f"; the data covers {lo} to {hi}, less than the {unit} requested, "
            "so this is the whole dataset"
        )
    return Caveat(kind="window", detail=detail, metrics={"last_n": float(win.last_n or 0)})


def _disclose_non_fact_rows(
    con: duckdb.DuckDBPyConnection,
    plan: BoundPlan,
    model: SemanticModel,
    event_time: str | None,
) -> Caveat | None:
    """When a revenue measure is computed WITHOUT the fact partition (a bare total or a
    grouping that is not a product), postage/fees/adjustments are still inside the
    number. Rather than hide that, sum the SAME measure over the excluded line types so
    the answer states exactly what non-product money it contains — a quantified, honest
    disclosure. Returns None when there is nothing to disclose: no fact partition in the
    model (e.g. raw UCI), or the partition was applied (a product ranking).
    """
    revenue = next((m for m in plan.measures if m.name in _REVENUE_MEASURES), None)
    if revenue is None:
        return None
    applied_cols = {f.column for f in plan.applied_filters}
    pd = next(
        (p for p in model.partition_dimensions if p.fact_value and p.column not in applied_cols),
        None,
    )
    if pd is None or pd.fact_value is None:
        return None

    col = _q(pd.column)
    clauses = [f"{col} IS NOT NULL", f"{col}::VARCHAR <> {_lit(pd.fact_value)}"]
    scope = _where(plan, event_time)  # the query's own filters, so the number matches
    if scope:
        clauses.insert(0, scope)
    where_sql = " AND ".join(clauses)
    result = con.execute(
        f"SELECT {col}::VARCHAR AS v, {revenue.sql} AS m FROM {_q(plan.table)} "
        f"WHERE {where_sql} GROUP BY {col} ORDER BY abs({revenue.sql}) DESC"
    ).fetchall()
    rows = [(str(v), float(m)) for v, m in result if m is not None and float(m) != 0.0]
    if not rows:
        return None

    total = sum(m for _, m in rows)
    top = rows[:3]
    listed = ", ".join(f"{v} {m:+,.2f}" for v, m in top)
    more = len(rows) - len(top)
    tail = f", +{more} more" if more > 0 else ""
    detail = (
        f"includes {len(rows)} non-{pd.fact_value} {pd.column} value(s) totalling "
        f"{total:+,.2f} ({listed}{tail}); restrict to {pd.column} = {pd.fact_value} "
        "to exclude them"
    )
    return Caveat(
        # A limitation of THIS answer, not an interpretation of the question: the total is
        # contaminated by non-fact money the user did not ask for and cannot rephrase away. It is
        # the fact partition going un-applied, which is exactly what the 'partition' kind names.
        kind="partition",
        detail=detail,
        metrics={"non_fact_total": round(total, 2), "non_fact_values": float(len(rows))},
    )


def _where(plan: BoundPlan, event_time: str | None) -> str:
    parts: list[str] = []
    for cf in [*plan.ir.categorical_filters, *plan.applied_filters]:
        parts.append(_categorical(cf))
    for nf in plan.ir.numeric_filters:
        parts.append(_numeric(nf))
    if plan.ir.time and event_time:
        parts.extend(_time(plan.ir.time, event_time, plan.table))
    if plan.require_complete_period_flag:
        parts.append(complete_period_clause(plan.require_complete_period_flag))
    return " AND ".join(parts)


def _describe_filters(plan: BoundPlan, event_time: str | None) -> list[str]:
    out: list[str] = []
    for cf in plan.ir.categorical_filters:
        out.append(_readable_categorical(cf))
    for cf in plan.applied_filters:
        out.append(_readable_categorical(cf) + "  [default: fact partition]")
    for nf in plan.ir.numeric_filters:
        out.append(_readable_numeric(nf))
    # Only report a time filter when a bound ACTUALLY constrains the rows. A bare grain carries
    # no bound - it is a grouping, compiled as a period column, not a filter - so an empty window
    # must not be announced as one (it would claim we narrowed the data when we did not).
    if plan.ir.time and event_time and _time(plan.ir.time, event_time, plan.table):
        win = plan.ir.time
        if win.named_period:
            out.append(f"time on {event_time}: {win.named_period}")
        elif win.last_n and win.grain:
            out.append(f"time on {event_time}: last {win.last_n} {win.grain}")
        else:
            bounds = [b for b in (f">= {win.start}" if win.start else "",
                                  f"< {win.end}" if win.end else "") if b]
            out.append(f"time on {event_time}: " + " ".join(bounds))
    if plan.require_complete_period_flag:
        out.append(f"only complete periods ({plan.require_complete_period_flag})  [default]")
    return out


def _readable_categorical(f: CategoricalFilter) -> str:
    verb = "not in" if f.op == "not_in" else "in"
    return f"{f.column} {verb} {f.values}"


def _readable_numeric(f: NumericFilter) -> str:
    if f.op == "between":
        return f"{f.column} between {f.value} and {f.value2}"
    return f"{f.column} {f.op} {f.value}"


def _explain(plan: BoundPlan, event_time: str | None) -> str:
    parts: list[str] = []
    if plan.ir.basket:
        return (
            "Ranked product pairs by how many distinct transactions contain both "
            "(product lines only, returns excluded)."
        )
    measures = ", ".join(m.name for m in plan.measures) or (
        "distinct count" if plan.ir.distinct_count_of else "count"
    )
    parts.append(f"Computed {measures}")
    if plan.group_by:
        parts.append(f"grouped by {', '.join(plan.group_by)}")
    filters = _describe_filters(plan, event_time)
    if filters:
        parts.append("filtered where " + "; ".join(filters))
    if plan.ir.period_comparison:
        pc = plan.ir.period_comparison
        parts.append(
            f"ranked the {pc.kind} of {measures} between two {pc.grain}s "
            "(each group's value in each period, then the delta)"
        )
    if plan.ir.top_k:
        parts.append(f"top {plan.ir.top_k.k} by {plan.ir.top_k.measure}")
    if plan.ir.share_of_total:
        parts.append("with each group's share of the total")
    return ". ".join(parts) + "."


def summary_value(answer: Answer) -> Any:
    """The single scalar for a non-grouped answer, else None."""
    if not answer.plan.group_by and len(answer.rows) == 1:
        row = answer.rows[0]
        return next(iter(row.values())) if len(row) == 1 else row
    return None
