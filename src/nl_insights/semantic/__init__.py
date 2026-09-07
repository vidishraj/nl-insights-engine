"""Semantic inference — Profile to a verified SemanticModel.

The LLM proposes roles over the profile's evidence; deterministic SQL verifiers test
every testable claim, discover the returns/partition/period structure, and resolve the
measure algebra. The result is the sole input to the interpreter.
"""

from .build import build_semantic_model
from .evidence import DatasetEvidence, build_evidence
from .model import (
    CategoricalValues,
    ColumnBinding,
    EntityLabel,
    MeasureBinding,
    PartitionDimension,
    PeriodCompleteness,
    ReturnsConvention,
    RoleClaim,
    SemanticModel,
    VerifierResult,
)
from .ontology import EntityKind, Role
from .proposer import build_request, propose_roles

__all__ = [
    "CategoricalValues",
    "ColumnBinding",
    "DatasetEvidence",
    "EntityKind",
    "EntityLabel",
    "MeasureBinding",
    "PartitionDimension",
    "PeriodCompleteness",
    "ReturnsConvention",
    "Role",
    "RoleClaim",
    "SemanticModel",
    "VerifierResult",
    "build_evidence",
    "build_request",
    "build_semantic_model",
    "propose_roles",
]
