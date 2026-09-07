"""The binder — deterministic, no LLM. The refusal spine.

THE invariant: there is no code path from a question to executed SQL that does not
pass here. The binder validates every IR node against the semantic model and emits
exactly one verdict, with five refusal classes each behind a deterministic detector:

1. missing concept   — a requested measure/role cannot be bound → REFUSE (cite it).
2. underdetermined    — revenue when returns exist / period-over-period on a partial
                        period → ANSWER_WITH_CAVEATS under a NAMED, documented default
                        (or CLARIFY when a follow-up genuinely changed nothing).
3. out-of-domain      — the plan computes nothing bindable → REFUSE before executing.
4. answerable-unreliable — a low-coverage entity is used → ANSWER_WITH_CAVEATS with the
                        coverage N/M taken from the profile, never from the LLM.
5. degenerate         — handled post-execution (see degenerate.py).
"""

from __future__ import annotations

from ..interpreter.ir import (
    Aggregation,
    CategoricalFilter,
    Grain,
    QueryIR,
    TimeWindow,
    named_period_grain,
)
from ..semantic.model import SemanticModel
from ..semantic.ontology import EntityKind, Role
from .verdict import BoundMeasure, BoundPlan, Caveat, Verdict, VerdictKind

# Coarseness order, so a grain can be compared to a named period's granularity.
_GRAIN_RANK = {"day": 0, "week": 1, "month": 2, "quarter": 3, "year": 4}


def _time_group_grain(ir: QueryIR, model: SemanticModel) -> Grain | None:
    """A grain becomes a GROUP BY - a per-period breakdown - when event_time is bound, UNLESS the
    window is a single named period of that same (or coarser) granularity, which is one bucket by
    definition. This reads what the fields mean: a named_period is a single bucket, a grain over a
    range or with no bound is a breakdown; range and last_n bounds apply as filters ON TOP of the
    grouping, they do not suppress it. A grain finer than the named period still groups within it.
    """
    if not (ir.time and ir.time.grain) or model.first_in_role(Role.EVENT_TIME) is None:
        return None
    npg = named_period_grain(ir.time.named_period)
    if npg is not None and _GRAIN_RANK[ir.time.grain] >= _GRAIN_RANK[npg]:
        return None  # the grain is the named period's own bucket (or coarser): a total
    return ir.time.grain

_COVERAGE_FLOOR = 0.98
_AGG_FUNCS = frozenset({"count", "sum", "avg", "min", "max"})


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _agg_bound_measure(agg: Aggregation) -> BoundMeasure:
    """The BoundMeasure for a validated generic aggregation. count(*) -> row_count (no
    column); sum/avg/min/max/count(col) -> func_col. The executor already compiles a
    BoundMeasure as ``{sql} AS {name}``, so a generic aggregation needs nothing new
    downstream."""
    if agg.func == "count" and agg.column is None:
        return BoundMeasure(name="row_count", sql="count(*)")
    assert agg.column is not None
    return BoundMeasure(name=f"{agg.func}_{agg.column}", sql=f"{agg.func}({_q(agg.column)})")


# The operator vocabularies the executor can compile. These mirror the IR's Literal types;
# the binder re-checks them so an IR built by code (model_construct, bypassing validation)
# with an unknown op REFUSES rather than reaching a compile site — a filter operator must
# never silently default to a branch (an inverted 'not_in' returns exactly what it excludes).
_CATEGORICAL_OPS = {"in", "not_in"}
_NUMERIC_OPS = {"gt", "gte", "lt", "lte", "eq", "between"}
_FREQUENCY_OPS = {"eq", "lte", "gte"}


_ROLE_VALUES = {r.value for r in Role}
_ENTITY_VALUES = {e.value for e in EntityKind}


def _resolve_ref(model: SemanticModel, ref: str) -> str | None:
    """Resolve an IR reference to a bound COLUMN. The interpreter may name a thing by its
    column, by its ROLE ('transaction_key'), or by its ENTITY KIND ('customer') — all
    three resolve here, so the binder and the executor share ONE contract (columns) rather
    than silently disagreeing about which namespace a value is in."""
    if any(b.column == ref and not b.refuted for b in model.bindings):
        return ref
    if ref in _ROLE_VALUES:
        found = model.first_in_role(Role(ref))
        return found.column if found else None
    if ref in _ENTITY_VALUES:
        cols = model.entity_key_columns(EntityKind(ref))
        return sorted(cols)[0] if cols else None
    return None


def _window_detail(win: TimeWindow) -> str | None:
    """A human statement of the effective time window, so the answer names it."""
    if win.named_period:
        return f"time window: {win.named_period}"
    if win.start or win.end:
        parts = []
        if win.start:
            parts.append(f"from {win.start} (inclusive)")
        if win.end:
            parts.append(f"to {win.end} (exclusive)")
        return "time window: " + " ".join(parts)
    if win.last_n and win.grain:
        return f"time window: last {win.last_n} {win.grain}"
    return None


def bind(model: SemanticModel, ir: QueryIR) -> Verdict:
    available = {m.name: m for m in model.measures if m.available and m.sql}
    bound_cols = {b.column for b in model.bindings if not b.refuted}
    dim_cols = {
        b.column
        for b in model.bindings
        if b.role in {Role.DIMENSION, Role.ENTITY_KEY} and not b.refuted
    }
    # Resolve IR references (frequency/distinct-count) from role/entity names to COLUMNS up
    # front, so the checks and the executor share one contract. A resolved reference is
    # rewritten into the IR; an unresolvable one is left as-is and refused below with an
    # accurate reason. Also normalise: naming basket_cooccurrence as a measure IS a basket.
    if "basket_cooccurrence" in ir.measures and not ir.basket:
        ir = ir.model_copy(update={"basket": True})
    if ir.frequency:
        e = _resolve_ref(model, ir.frequency.entity)
        c = _resolve_ref(model, ir.frequency.count_of)
        if e and c:
            ir = ir.model_copy(
                update={"frequency": ir.frequency.model_copy(update={"entity": e, "count_of": c})}
            )
    if ir.distinct_count_of:
        d = _resolve_ref(model, ir.distinct_count_of)
        if d:
            ir = ir.model_copy(update={"distinct_count_of": d})

    # underdetermined by the interpreter: it flagged a concept it could not express (a
    # quantity that needs data this file lacks). Refuse NAMING it — checked before
    # out-of-domain so the reason cites the concept, and do NOT let a substituted measure
    # be presented as the answer.
    if ir.unmet_concepts:
        concepts = ", ".join(ir.unmet_concepts)
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason=f"Cannot compute: {concepts} — not available in this dataset.",
            evidence={"unmet": ir.unmet_concepts, "available_measures": sorted(available)},
        )

    # a requested BREAKDOWN (or filter) names an attribute this dataset does not have (an
    # attribute with no matching column). The interpreter must surface it here rather than
    # silently drop the grouping and return an ungrouped scalar — a differently-shaped answer
    # to a different question. Turn it into a CLARIFY that offers the dimensions that DO
    # exist, so the user chooses instead of the system inventing. Checked before out-of-domain
    # so a bare unmet breakdown clarifies (names the gap) rather than refusing generically.
    if ir.unmet_dimensions:
        available_dims = sorted(dim_cols)
        dims_text = ", ".join(available_dims) if available_dims else "(no grouping dimensions)"
        # ENUMERATE what exists rather than assert what does not — a message that lists the
        # available groupings cannot falsely claim a column is absent. Distinguish a column
        # that IS bound but is not a grouping (e.g. a flag like 'survived') from one that
        # genuinely does not exist: the old wording ("no X column") was factually false for a
        # bound flag, asserting non-existence when it meant non-groupability.
        bound_not_dim = [a for a in ir.unmet_dimensions if a in bound_cols and a not in dim_cols]
        if bound_not_dim:
            question = (
                f"{', '.join(bound_not_dim)} exists but is not a grouping dimension here. "
                f"Group by one of: {dims_text}; or ask for the overall total."
            )
        else:
            asked = ", ".join(ir.unmet_dimensions)
            question = (
                f"'{asked}' is not an available grouping in this dataset. "
                f"Group by one of: {dims_text}; or ask for the overall total."
            )
        return Verdict(
            kind=VerdictKind.CLARIFY,
            clarify_question=question,
            options=[*available_dims, "overall total (no grouping)"],
            evidence={
                "unmet_dimensions": ir.unmet_dimensions,
                "available_dimensions": available_dims,
                "bound_but_not_groupable": bound_not_dim,
            },
        )

    # (3) out-of-domain: the plan asks for nothing computable.
    if not (ir.measures or ir.distinct_count_of or ir.frequency or ir.basket or ir.aggregations):
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason="The question does not map to any measure or count over this dataset.",
            evidence={"interpreter_notes": ir.notes, "available_measures": sorted(available)},
        )

    # generic aggregation integrity: count(*) always applies (no column); count(col) needs a
    # real column; sum/avg/min/max need a NUMERIC one — validated against model.numeric_columns,
    # which covers a numeric the proposer gave a non-measure role ('ignore'), so an ordinary
    # numeric (a fare, a life expectancy) can be aggregated. An un-numeric or unknown target
    # REFUSES, never silently produces a SQL error or a wrong number.
    all_columns = {b.column for b in model.bindings} | set(model.numeric_columns)
    numeric_cols = set(model.numeric_columns)
    agg_measures: list[BoundMeasure] = []
    for agg in ir.aggregations:
        if agg.func not in _AGG_FUNCS:
            return Verdict(
                kind=VerdictKind.REFUSE,
                reason=f"Unsupported aggregation: {agg.func!r}.",
                evidence={"bad_func": agg.func, "supported": sorted(_AGG_FUNCS)},
            )
        if agg.func == "count" and agg.column is None:
            agg_measures.append(_agg_bound_measure(agg))
            continue
        if agg.column is None or agg.column not in all_columns:
            return Verdict(
                kind=VerdictKind.REFUSE,
                reason=f"Cannot aggregate {agg.column!r}: it is not a column in this dataset.",
                evidence={"aggregation": agg.model_dump(), "columns": sorted(all_columns)},
            )
        if agg.func != "count" and agg.column not in numeric_cols:
            return Verdict(
                kind=VerdictKind.REFUSE,
                reason=f"Cannot {agg.func} {agg.column!r}: it is not a numeric column.",
                evidence={"aggregation": agg.model_dump(), "numeric_columns": sorted(numeric_cols)},
            )
        agg_measures.append(_agg_bound_measure(agg))

    # (C) competing money columns: if the query asks for a REVENUE measure and the model has
    # MORE THAN ONE monetary_amount, which column is 'the' revenue is ambiguous — the supermarket
    # trap, where a cost column verifies against qty×rate while the true sales total, carrying
    # tax, does not. CLARIFY listing the candidates rather than sum one silently. The trigger is
    # strictly '> 1 monetary_amount proposal', NOT 'several money-ish columns': a file with a
    # SINGLE amount column never clarifies, so the flagship product ranking is untouched.
    # Checked before missing-concept so a >1-amount revenue query clarifies rather than
    # refusing on availability.
    revenue_asked = any(m in {"net_revenue", "gross_revenue"} for m in ir.measures)
    amount_cols = sorted(
        b.column for b in model.bindings if b.role == Role.MONETARY_AMOUNT and not b.refuted
    )
    if revenue_asked and len(amount_cols) > 1:
        return Verdict(
            kind=VerdictKind.CLARIFY,
            clarify_question=(
                f"Several columns could be the revenue amount: {', '.join(amount_cols)}. "
                "Which should I sum? (Or ask for one directly, e.g. its total.)"
            ),
            options=amount_cols,
            evidence={"competing_amounts": amount_cols},
        )

    # (1) missing concept: a requested measure/analysis has no verified binding. A measure
    # is present if it is AVAILABLE (basket_cooccurrence is available with sql=None — it is
    # a self-join — so it is checked by the flag, not the scalar-sql dict).
    available_names = {m.name for m in model.measures if m.available}
    missing: list[str] = [m for m in ir.measures if m not in available_names]
    if ir.basket and "basket_cooccurrence" not in available_names:
        missing.append("basket_cooccurrence")
    if ir.distinct_count_of and ir.distinct_count_of not in bound_cols:
        missing.append(f"a distinct count of {ir.distinct_count_of!r}")
    if ir.frequency:
        if ir.frequency.count_of not in bound_cols:
            missing.append("a verified transaction key to count purchases")
        if ir.frequency.entity not in bound_cols:
            missing.append("an entity to count purchases per")
    if missing:
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason=f"Cannot compute: {', '.join(missing)}.",
            evidence={
                "missing": missing,
                "available_measures": sorted(available),
                "hint": "a required role is unbound or was refuted by a verifier",
            },
        )

    # reference integrity: every dimension/filter names a real, non-refuted column.
    unbound = [g for g in ir.group_by if g not in dim_cols and g not in bound_cols]
    unbound += [f.column for f in ir.categorical_filters if f.column not in bound_cols]
    unbound += [f.column for f in ir.numeric_filters if f.column not in bound_cols]
    if unbound:
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason=f"Unknown or unusable columns: {', '.join(sorted(set(unbound)))}.",
            evidence={"unbound": sorted(set(unbound)), "bound_columns": sorted(bound_cols)},
        )
    if (ir.time or ir.period_comparison) and model.first_in_role(Role.EVENT_TIME) is None:
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason="A time filter/comparison was requested but no event_time is bound.",
            evidence={"needs": "event_time"},
        )

    # operator integrity: every filter operator must be one the executor can compile. The IR
    # Literal already constrains what the model can emit; this catches an IR constructed in
    # code that bypassed validation, so an unknown op REFUSES instead of silently selecting a
    # default (opposite) branch at a compile site.
    bad_ops: list[str] = [f.op for f in ir.categorical_filters if f.op not in _CATEGORICAL_OPS]
    bad_ops += [f.op for f in ir.numeric_filters if f.op not in _NUMERIC_OPS]
    if ir.frequency and ir.frequency.op not in _FREQUENCY_OPS:
        bad_ops.append(ir.frequency.op)
    if bad_ops:
        return Verdict(
            kind=VerdictKind.REFUSE,
            reason=f"Unsupported filter operator(s): {', '.join(sorted(set(bad_ops)))}.",
            evidence={
                "bad_operators": sorted(set(bad_ops)),
                "categorical_ops": sorted(_CATEGORICAL_OPS),
                "numeric_ops": sorted(_NUMERIC_OPS),
                "frequency_ops": sorted(_FREQUENCY_OPS),
            },
        )

    # value integrity: a filter VALUE must exist in its column, not just the column NAME. The
    # interpreter can invent a value the data never had (filtering a status column on a value
    # absent from that column), which otherwise executes to a confident, uncaveated ZERO. We
    # can only be CERTAIN a value is absent when the observed set is EXHAUSTIVE (a sample proves
    # nothing about absence), so we CLARIFY only then — and case-insensitively, so a casing
    # difference is not mistaken for an invented value. A value missing from a NON-exhaustive
    # (sampled) column is left to the executor's zero-match guard.
    for cf in ir.categorical_filters:
        cv = model.categorical_values.get(cf.column)
        if cv is None or not cv.exhaustive:
            continue
        known = {v.casefold() for v in cv.values}
        invented = [v for v in cf.values if v.casefold() not in known]
        if invented:
            return Verdict(
                kind=VerdictKind.CLARIFY,
                clarify_question=(
                    f"{cf.column} has no value {', '.join(repr(v) for v in invented)}. "
                    f"Its values are {', '.join(cv.values)} — which did you mean?"
                ),
                options=cv.values,
                evidence={"column": cf.column, "invented": invented, "actual_values": cv.values},
            )

    # reference integrity of the RANKING node: an ORDER BY target must be a column the
    # query actually computes — a selected measure or a grouping (or the distinct/share
    # output). Ranking by anything else compiles to a SQL error, so it is a refusal, not
    # an answer. (basket/frequency have a fixed internal ordering and skip this.)
    if ir.top_k and not ir.basket and not ir.frequency:
        order_targets = set(ir.measures) | set(ir.group_by) | {m.name for m in agg_measures}
        if ir.distinct_count_of:
            order_targets.add("distinct_count")
        if ir.share_of_total:
            order_targets.add("share")
        if ir.period_comparison:
            # a growth comparison ranks by the computed delta; the measure name is also
            # accepted (the executor always ranks by growth for this shape).
            order_targets.add("growth")
        if ir.top_k.measure not in order_targets:
            return Verdict(
                kind=VerdictKind.REFUSE,
                reason=(
                    f"Cannot rank by {ir.top_k.measure!r}: it is not a measure or grouping "
                    "computed by this query."
                ),
                evidence={
                    "order_by": ir.top_k.measure,
                    "rankable": sorted(order_targets),
                },
            )

    # CLARIFY: a follow-up that changed nothing (flagged by the merge).
    if any("clarification" in n.lower() for n in ir.notes):
        return Verdict(
            kind=VerdictKind.CLARIFY,
            clarify_question="What would you like to change from the previous result?",
            options=["a different time period", "a different measure", "a different grouping"],
        )

    # --- ANSWERABLE path: resolve the plan and collect caveats ---
    caveats: list[Caveat] = []
    applied_filters: list[CategoricalFilter] = []

    # disclose the date interpretation on any time query — the resolver decided the
    # day/month order at ingest, and a system that discloses its assumptions must say so
    # rather than swallow it.
    if ir.time or ir.period_comparison:
        et = model.first_in_role(Role.EVENT_TIME)
        note = model.date_assumptions.get(et.column) if et else None
        if note:
            caveats.append(Caveat(kind="assumption", detail=note))
    # state the actual window used — a time answer that doesn't name its window is hard to
    # trust, and it makes any off-by-one visible instead of silent.
    if ir.time:
        window = _window_detail(ir.time)
        if window:
            caveats.append(Caveat(kind="window", detail=window))

    # (B) revenue summed from a single UNVERIFIED amount — disclose it in PROSE the user can act
    # on: name the column, and say the check could not be RUN (there is no quantity×rate to check
    # it against), which is different from the check FAILING. This disclosure IS the safety
    # property that makes 'use the one money column as reported' honest rather than a silent guess.
    if revenue_asked and len(amount_cols) == 1:
        amount_binding = next(b for b in model.bindings if b.column == amount_cols[0])
        verified = any(
            v.name == "stored_amount_identity" and v.passed for v in amount_binding.verifiers
        )
        has_qty_rate = bool(
            model.first_in_role(Role.ADDITIVE_QUANTITY) and model.first_in_role(Role.MONETARY_RATE)
        )
        if not verified and not has_qty_rate:
            caveats.append(
                Caveat(
                    kind="assumption",
                    detail=(
                        f"{amount_cols[0]} is summed AS REPORTED: there are no quantity and rate "
                        "columns, so it could not be checked against quantity × rate — the check "
                        "could not be run, which is different from it failing"
                    ),
                )
            )

    # (2) underdetermined — revenue when returns exist: answer NET under a named default.
    if "net_revenue" in ir.measures and model.returns.kind not in {"none", ""}:
        caveats.append(
            Caveat(
                kind="assumption",
                detail=(
                    "'revenue' is reported NET of returns "
                    f"({model.returns.kind}); ask for gross_revenue to exclude them"
                ),
            )
        )

    # partition default: the partition is a property of the FACT TABLE and the PRODUCT
    # ENTITY being ranked — NOT of the group-by column. A query that ranks products must
    # exclude non-product rows (postage, adjustments) or 'top products by revenue' is
    # wrong; and "ranks products" means grouping by ANY product identifier — the entity
    # key OR a description that labels it (grouping by 'description' is the natural human
    # phrasing and must behave identically to grouping by the code). A bare revenue total
    # is deliberately left un-partitioned — postage IS revenue — but that reading is
    # DISCLOSED below rather than applied silently, since silent-no-filter is the worst
    # of the three options.
    # The partition is a property of the FACT TABLE and the PRODUCT ENTITY ranked, NOT
    # of the group-by column: a query that ranks products (grouping by the entity key OR
    # any label for it) must exclude non-product rows or 'top products by revenue' is
    # wrong. A bare total / non-product grouping is deliberately left un-partitioned —
    # postage IS revenue — but that reading is DISCLOSED (and QUANTIFIED) by the executor,
    # which has the data, not silently applied. Silent-no-filter is the worst option.
    ranks_products = bool(model.product_identifier_columns() & set(ir.group_by))
    if ranks_products:
        for pd in model.partition_dimensions:
            if pd.fact_value:
                applied_filters.append(
                    CategoricalFilter(column=pd.column, op="in", values=[pd.fact_value])
                )
                caveats.append(
                    Caveat(
                        kind="partition",
                        detail=(
                            f"restricted to {pd.column} = {pd.fact_value} (the product lines); "
                            "non-product rows are excluded"
                        ),
                    )
                )

    # (2) underdetermined — period comparison over a partial period: exclude incomplete.
    require_complete = None
    if ir.period_comparison and model.period_completeness:
        require_complete = model.period_completeness[0].flag_column
        caveats.append(
            Caveat(
                kind="period",
                detail=f"incomplete periods excluded via {require_complete}",
            )
        )

    # (4) answerable-but-unreliable: low-coverage entity used → quantified caveat.
    used = set(ir.group_by) | ({ir.distinct_count_of} if ir.distinct_count_of else set())
    if ir.frequency:
        used |= {ir.frequency.entity, ir.frequency.count_of}
    for col in used:
        cov = model.coverage.get(col)
        if cov is not None and cov < _COVERAGE_FLOOR:
            n = round(cov * model.row_count)
            caveats.append(
                Caveat(
                    kind="coverage",
                    detail=(
                        f"computed over the {cov:.1%} of rows that carry {col} "
                        f"({n}/{model.row_count})"
                    ),
                    metrics={"coverage": cov, "rows": float(n), "total": float(model.row_count)},
                )
            )

    plan = BoundPlan(
        ir=ir,
        table=model.table,
        # only scalar-sql measures become BoundMeasures; basket_cooccurrence (sql=None) is
        # executed by the basket self-join path, not as an aggregate here.
        measures=[
            BoundMeasure(name=n, sql=available[n].sql)  # type: ignore[arg-type]
            for n in ir.measures
            if n in available
        ]
        + agg_measures,
        group_by=ir.group_by,
        applied_filters=applied_filters,
        require_complete_period_flag=require_complete,
        time_group_grain=_time_group_grain(ir, model),
        coverage=model.coverage,
    )

    # On the answerable path the ANSWERABLE vs ANSWER_WITH_CAVEATS split is PROVISIONAL: it is
    # finalised from the finished answer by Answer.verdict_kind(), because a caveat can be born at
    # execute time (the non-fact partition disclosure) and this bind-time view cannot see it. Do
    # NOT serialise this kind for an answered query - the pipeline replaces it post-execute. It is
    # kept as an honest first read (based on the binder's own caveats: a fact partition, an
    # excluded period, low coverage; assumptions never make an answer 'with caveats').
    has_real_caveat = any(c.kind != "assumption" for c in caveats)
    kind = VerdictKind.ANSWER_WITH_CAVEATS if has_real_caveat else VerdictKind.ANSWERABLE
    return Verdict(kind=kind, plan=plan, caveats=caveats)
