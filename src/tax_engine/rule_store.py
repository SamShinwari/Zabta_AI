from decimal import Decimal
from typing import Optional

from .models import TaxRule


class TaxRuleStore:
    """
    Stores tax rules used by the tax engine.

    The store is intentionally simple for the MVP.
    Later it can be replaced by a database or RAG-backed
    rule repository.
    """

    def __init__(self):
        self._rules: dict[str, TaxRule] = {}

    def add_rule(self, rule: TaxRule) -> None:
        self._rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> Optional[TaxRule]:
        return self._rules.get(rule_id)

    def all_rules(self) -> list[TaxRule]:
        return list(self._rules.values())

    def find_rate(self, rate: Decimal) -> Optional[TaxRule]:
        for rule in self._rules.values():
            if rule.tax_rate == rate:
                return rule

        return None


def create_default_rule_store() -> TaxRuleStore:
    """
    Create the initial MVP rule store.

    IMPORTANT:
    Legal tax rates should eventually come from the
    curated FBR knowledge base rather than being treated
    as permanent hard-coded values.
    """

    store = TaxRuleStore()

    return store
