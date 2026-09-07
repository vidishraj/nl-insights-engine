"""The fixed, generic ontology of transactional roles.

Nothing here is dataset-specific — these are the roles a *transactional* table can
play, independent of column names. The LLM proposes which column fills which role;
the verifiers (SQL) then test the proposals against the data. Because the ontology is
closed and generic, the same model works on a file whose columns carry one naming
convention and on one whose header is missing entirely (columns then seen positionally).
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    EVENT_TIME = "event_time"  # when the transaction happened
    TRANSACTION_KEY = "transaction_key"  # groups line items into one transaction
    ENTITY_KEY = "entity_key"  # identifies an entity (customer, product, …); see EntityKind
    ADDITIVE_QUANTITY = "additive_quantity"  # counts that sum meaningfully (units sold)
    MONETARY_RATE = "monetary_rate"  # per-unit price; NOT additive across a transaction
    MONETARY_AMOUNT = "monetary_amount"  # an extended amount (quantity × rate); additive
    DISCOUNT = "discount"  # a reduction applied to an amount
    DIMENSION = "dimension"  # a categorical to slice by (country, category, …)
    FLAG = "flag"  # a boolean / indicator column
    DESCRIPTION = "description"  # free text describing an entity
    IGNORE = "ignore"  # operator notes, opaque ids with no analytical role


class EntityKind(StrEnum):
    CUSTOMER = "customer"
    PRODUCT = "product"
    LOCATION = "location"
    OTHER = "other"


# Additivity class per role: can a column in this role be summed across the grain of a
# query? A rate cannot (summing unit prices is nonsense); a quantity or amount can.
ADDITIVE_ROLES = frozenset({Role.ADDITIVE_QUANTITY, Role.MONETARY_AMOUNT, Role.DISCOUNT})
NON_ADDITIVE_ROLES = frozenset({Role.MONETARY_RATE})


def is_additive(role: Role) -> bool:
    return role in ADDITIVE_ROLES
