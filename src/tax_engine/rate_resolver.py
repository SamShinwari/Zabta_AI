from decimal import Decimal
from typing import Optional

from .rule_store import TaxRuleStore


def resolve_tax_rate(
    declared_rate,
    rule_store: TaxRuleStore,
) -> dict:
    """
    Resolve a declared tax rate against the configured rules.

    This function does not decide whether a rate is legally
    applicable to a particular product. That requires
    additional rule/context resolution.
    """

    if declared_rate is None:
        return {
            "rate": None,
            "valid": False,
            "rule": None,
            "reason": "No tax rate was declared.",
        }

    rate = Decimal(str(declared_rate))

    rule = rule_store.find_rate(rate)

    if rule is None:
        return {
            "rate": rate,
            "valid": False,
            "rule": None,
            "reason": (
                "Declared tax rate was not found "
                "in the configured rule store."
            ),
        }

    return {
        "rate": rate,
        "valid": True,
        "rule": rule,
        "reason": "Declared tax rate matched a configured rule.",
    }
