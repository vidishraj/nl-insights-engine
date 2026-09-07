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
from ..executor import execute
from ..interpreter.ir import QueryIR
from ..semantic.model import SemanticModel


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


def run_case(model: SemanticModel, con: duckdb.DuckDBPyConnection, case: GoldenCase) -> CaseResult:
    verdict = bind(model, case.ir)
    actual = classify(verdict.kind)
    detail = verdict.reason or ""
    if actual is Expected.ANSWER and verdict.plan is not None:
        # Execute so an 'answer' that would blow up at run time counts as a failure here.
        answer = execute(con, verdict.plan, model, verdict.caveats)
        detail = f"{len(answer.rows)} rows"
    passed = actual is case.expected
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
