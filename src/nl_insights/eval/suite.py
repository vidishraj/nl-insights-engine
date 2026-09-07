"""Golden verdict suites and the refusal confusion matrix — how we know it works.

A golden case pairs a question with the plan the interpreter is expected to produce and
the verdict CLASS it must earn (answer / refuse / clarify). The runner binds each case
and, where an answer is expected, executes it. It reports a confusion matrix and, above
all, the FALSE-ANSWER RATE: the fraction of should-not-answer cases where the system
invented an answer anyway. That is the number 'stop inventing answers' is about; a
false refusal is the cheaper error and is reported separately.

Cases are expressed at the IR level so the suite is deterministic and hermetic (the
LLM, the one non-deterministic part, is tested separately under record/replay). This
also means the same case can be re-bound after ablating a column to prove the verdict
is driven by the DATA, not the question.
"""

from __future__ import annotations

from enum import StrEnum

import duckdb
from pydantic import BaseModel

from ..binder import VerdictKind, bind
from ..executor import Answer, execute
from ..interpreter.ir import QueryIR, named_period_grain
from ..semantic.model import SemanticModel

# Coarseness order, so a requested grain can be compared to a named period's granularity.
_GRAIN_RANK = {"day": 0, "week": 1, "month": 2, "quarter": 3, "year": 4}


class Expected(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    CLARIFY = "clarify"


_ANSWER_KINDS = {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}


def classify(kind: VerdictKind) -> Expected:
    if kind in _ANSWER_KINDS:
        return Expected.ANSWER
    if kind is VerdictKind.CLARIFY:
        return Expected.CLARIFY
    return Expected.REFUSE


class GoldenCase(BaseModel):
    name: str
    question: str  # the natural-language question (documentation + paraphrase anchor)
    ir: QueryIR  # the plan the interpreter is expected to produce for it
    expected: Expected
    # Columns the answer genuinely depends on — dropping any must flip it to a refusal.
    supporting_columns: list[str] = []
    paraphrases: list[str] = []  # hostile rephrasings (used by the replay-backed suite)


class CaseResult(BaseModel):
    name: str
    expected: Expected
    actual: Expected
    passed: bool
    false_answer: bool  # invented an answer where a refusal/clarify was correct
    detail: str = ""


class SuiteReport(BaseModel):
    dataset: str
    results: list[CaseResult]
    matrix: dict[str, dict[str, int]]  # expected -> actual -> count
    false_answer_rate: float  # invented answers / should-not-answer cases (THE metric)
    false_refusal_rate: float  # wrongly refused / should-answer cases (the cheaper error)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)


def _time_grain_honored(answer: Answer) -> bool:
    """A requested time grain is honored two ways only: it became a compiled per-period grouping
    (plan.time_group_grain is set), or it collapsed into a single named period of that same or
    coarser granularity, which is one bucket by definition. Anything else is a dropped grain."""
    ir = answer.plan.ir
    if answer.plan.time_group_grain is not None:
        return True
    if not (ir.time and ir.time.grain):
        return True  # no grain was requested; nothing to honor
    npg = named_period_grain(ir.time.named_period)
    return npg is not None and _GRAIN_RANK[ir.time.grain] >= _GRAIN_RANK[npg]


def coherence_violations(kind: VerdictKind, answer: Answer) -> list[str]:
    """Internal-coherence invariants EVERY produced answer must satisfy, independent of dataset
    or question - the checks a verdict-class assertion cannot make, and which constrain answers
    nobody has written a case for yet:

      1. an ANSWER_WITH_CAVEATS verdict must disclose an actual CAVEAT. A caveat says what is
         limited about THIS answer; an assumption only says how we read the data (dates are
         day-first), and almost every answer carries one. Accepting an assumption as disclosure
         makes this blind to a caveats verdict that hides a limitation, so it is caveats only.
      1b. and its converse, so the implication is closed both ways: an answer that DOES carry a
         real (non-assumption) caveat must be ANSWER_WITH_CAVEATS, never ANSWERABLE. Without the
         converse, an ANSWERABLE answer could carry 'includes 59,256 of non-product money' and
         claim no limitation - the same incoherence as 1, pointing the other way.
      2. an announced filter must correspond to a real WHERE. An empty window is not a filter and
         must not be reported as one.
      3. a requested grouping that was dropped must be recorded as an unmet dimension, never
         silently returned as a total - WHEREVER the request is represented. A categorical
         breakdown lives in ir.group_by; a time grain lives in ir.time, NOT group_by, which is
         exactly why a group_by-only check was blind to the by-period defect that motivated this.
    """
    ir = answer.plan.ir
    out: list[str] = []
    if kind is VerdictKind.ANSWER_WITH_CAVEATS and not answer.caveats:
        out.append("answer_with_caveats but no caveat is disclosed")
    if answer.caveats and kind is not VerdictKind.ANSWER_WITH_CAVEATS:
        out.append("a real caveat is present but the verdict is not answer_with_caveats")
    if answer.filters_applied and "WHERE" not in answer.sql.upper():
        out.append("filters_applied announced but no WHERE clause constrains the query")
    unrecorded = (set(ir.group_by) - set(answer.plan.group_by)) - set(ir.unmet_dimensions)
    if unrecorded:
        out.append(f"requested grouping dropped without recording as unmet: {sorted(unrecorded)}")
    grain = ir.time.grain if ir.time else None
    if grain is not None and not _time_grain_honored(answer) and not ir.unmet_dimensions:
        out.append(
            f"requested time grain '{grain}' neither grouped, bucketed, "
            "nor recorded as an unmet dimension"
        )
    return out


def run_case(model: SemanticModel, con: duckdb.DuckDBPyConnection, case: GoldenCase) -> CaseResult:
    verdict = bind(model, case.ir)
    actual = classify(verdict.kind)
    detail = verdict.reason or ""
    coherent = True
    if actual is Expected.ANSWER and verdict.plan is not None:
        # Execute so an 'answer' that would blow up at run time counts as a failure here, and
        # assert the answer is internally coherent - not merely of the expected verdict class. The
        # class is the FINALISED one from the finished answer (the same class every consumer
        # reports), never the provisional bind-time kind.
        answer = execute(con, verdict.plan, model, verdict.caveats)
        violations = coherence_violations(answer.verdict_kind(), answer)
        coherent = not violations
        detail = f"{len(answer.rows)} rows" + (
            "" if coherent else "; INCOHERENT: " + "; ".join(violations)
        )
    passed = actual is case.expected and coherent
    false_answer = (
        case.expected in {Expected.REFUSE, Expected.CLARIFY} and actual is Expected.ANSWER
    )
    return CaseResult(
        name=case.name,
        expected=case.expected,
        actual=actual,
        passed=passed,
        false_answer=false_answer,
        detail=detail,
    )


def run_suite(
    dataset: str,
    model: SemanticModel,
    con: duckdb.DuckDBPyConnection,
    cases: list[GoldenCase],
) -> SuiteReport:
    results = [run_case(model, con, c) for c in cases]
    labels = [Expected.ANSWER, Expected.REFUSE, Expected.CLARIFY]
    counts: dict[Expected, dict[Expected, int]] = {e: dict.fromkeys(labels, 0) for e in labels}
    for r in results:
        counts[r.expected][r.actual] += 1

    should_not_answer = sum(1 for r in results if r.expected is not Expected.ANSWER)
    invented = sum(1 for r in results if r.false_answer)
    should_answer = sum(1 for r in results if r.expected is Expected.ANSWER)
    wrongly_refused = sum(
        1 for r in results if r.expected is Expected.ANSWER and r.actual is not Expected.ANSWER
    )
    return SuiteReport(
        dataset=dataset,
        results=results,
        matrix={e.value: {a.value: counts[e][a] for a in labels} for e in labels},
        false_answer_rate=(invented / should_not_answer) if should_not_answer else 0.0,
        false_refusal_rate=(wrongly_refused / should_answer) if should_answer else 0.0,
    )


def format_report(report: SuiteReport) -> str:
    """A compact, human-readable rendering for the runner script."""
    labels = [Expected.ANSWER, Expected.REFUSE, Expected.CLARIFY]
    lines = [f"# {report.dataset}", "confusion matrix (rows=expected, cols=actual):"]
    header = "            " + "".join(f"{a.value:>9}" for a in labels)
    lines.append(header)
    for e in labels:
        row = "".join(f"{report.matrix[e.value][a.value]:>9}" for a in labels)
        lines.append(f"{e.value:>11} {row}")
    lines.append(
        f"false-answer rate : {report.false_answer_rate:.3f}  (invented answers — THE metric)"
    )
    lines.append(f"false-refusal rate: {report.false_refusal_rate:.3f}  (cheaper error)")
    for r in report.results:
        mark = "ok " if r.passed else "XX "
        lines.append(f"  {mark}{r.name}: expected {r.expected}, got {r.actual}  {r.detail}")
    return "\n".join(lines)
