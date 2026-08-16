from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class TaxRule:
    """
    Represents one tax rule used by Zabta.

    Tax rules should be configurable rather than
    hard-coded inside calculation functions.
    """

    rule_id: str
    description: str
    tax_rate: Decimal
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    source_document: Optional[str] = None
    source_reference: Optional[str] = None


@dataclass
class TaxResult:
    """
    Result of tax validation/calculation.
    """

    taxable_amount: Decimal
    applicable_rate: Decimal
    expected_tax: Decimal
    declared_tax: Optional[Decimal]
    difference: Optional[Decimal]
    is_valid: bool
    rule_id: Optional[str] = None
    explanation: str = ""
    source_document: Optional[str] = None
    source_reference: Optional[str] = None
