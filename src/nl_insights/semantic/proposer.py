"""Role assignment — the first (and only schema-facing) LLM call.

The model receives an evidence pack per column and proposes a role from the FIXED
ontology, with a confidence and reasons. It is constrained to structured output (a
forced schema), never free text. The prompt is deliberately dataset-agnostic: it names
no dataset, no column, no value — it describes the generic roles and asks the model to
reason from the evidence, treating a column's name as one weak signal among the stats,
samples, and functional dependencies. The proposal is then handed to the verifiers,
which test it in SQL; the model is never trusted on its own say-so.
"""

from __future__ import annotations

from ..provider import LLMProvider, LLMRequest
from .evidence import DatasetEvidence
from .model import RoleClaim
from .ontology import EntityKind, Role

_ROLES = [r.value for r in Role]
_ENTITIES = [e.value for e in EntityKind]

SYSTEM = (
    "You assign each column of a transactional table exactly one ROLE from a fixed, "
    "generic ontology. You are given, per column, only statistical evidence: semantic "
    "type, null rate, cardinality, uniqueness, numeric sign structure, top values, a "
    "sample of values, and the functional dependencies it participates in (with a "
    "strength and a support count). A column's NAME may be present but is only a weak "
    "hint — the same table may arrive with no header at all, so decide from the "
    "evidence. Roles:\n"
    "- event_time: when a transaction occurred (a date/timestamp).\n"
    "- transaction_key: groups line items into one transaction; high cardinality but "
    "recurring, and it strongly determines the event_time (see the dependencies).\n"
    "- entity_key: identifies an entity; set 'entity' to customer, product, location, "
    "or other.\n"
    "- additive_quantity: a count that sums meaningfully (integer-dominant; may be "
    "negative for returns).\n"
    "- monetary_rate: a per-unit price (decimal-bearing; NOT additive across a "
    "transaction).\n"
    "- monetary_amount: an extended amount, i.e. quantity times rate (additive).\n"
    "- discount: a reduction applied to an amount.\n"
    "- dimension: a categorical to slice by (country, category, ...).\n"
    "- flag: a boolean/indicator column.\n"
    "- description: free text describing an entity.\n"
    "- ignore: opaque notes/ids with no analytical role.\n"
    "Give a confidence in [0,1] and short evidence-grounded reasons. Every column gets "
    "exactly one claim."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "role": {"type": "string", "enum": _ROLES},
                    "entity": {"type": ["string", "null"], "enum": [*_ENTITIES, None]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["column", "role", "confidence", "reasons"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def build_request(evidence: DatasetEvidence, model: str) -> LLMRequest:
    # Exclude the dataset_id: a file's NAME is identity, not evidence about its semantics,
    # and we claim not to lean on names — feeding it to the proposer would quietly break
    # that. It also makes the request (and thus the replay cache key) a pure function of
    # the file's structure and statistics, so the same bytes replay under any upload name.
    prompt = "Assign a role to every column. Evidence (JSON):\n" + evidence.model_dump_json(
        indent=None, exclude={"dataset_id"}
    )
    return LLMRequest(model=model, system=SYSTEM, prompt=prompt, schema=RESPONSE_SCHEMA)


def propose_roles(provider: LLMProvider, evidence: DatasetEvidence, model: str) -> list[RoleClaim]:
    response = provider.complete(build_request(evidence, model))
    known = {c.name for c in evidence.columns}
    claims: list[RoleClaim] = []
    for raw in response.get("claims", []):
        if raw.get("column") in known:
            claims.append(RoleClaim.model_validate(raw))
    return claims
