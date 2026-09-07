"""Eval — a NAMED component that answers 'how do you know it works'.

It bundles: golden verdict suites with a refusal confusion matrix and a false-answer
rate (the metric that matters); metamorphic ablation that proves refusals are driven by
the DATA, not the phrasing; header-stripping that proves column names are only a weak
signal; and an anti-hardcoding lint that makes 'nothing is dataset-specific' falsifiable.
"""

from .ablation import AblationResult, ablate_csv, run_ablation
from .generality import strip_headers
from .lint import FORBIDDEN_TOKENS, Violation, scan
from .suite import (
    CaseResult,
    Expected,
    GoldenCase,
    SuiteReport,
    classify,
    format_report,
    run_case,
    run_suite,
)

__all__ = [
    "AblationResult",
    "CaseResult",
    "Expected",
    "FORBIDDEN_TOKENS",
    "GoldenCase",
    "SuiteReport",
    "Violation",
    "ablate_csv",
    "classify",
    "format_report",
    "run_ablation",
    "run_case",
    "run_suite",
    "scan",
    "strip_headers",
]
