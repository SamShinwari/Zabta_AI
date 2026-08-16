from decimal import Decimal

from .tax_calculator import calculate_tax, calculate_difference
from .rate_resolver import resolve_tax_rate
from .rule_store import TaxRuleStore
from .models import TaxResult


def validate_tax(
    taxable_amount,
    declared_rate,
    declared_tax,
    rule_store: TaxRuleStore,
    tolerance=Decimal("0.01"),
) -> TaxResult:

    rate_result = resolve_tax_rate(
        declared_rate,
        rule_store,
    )

    if not rate_result["valid"]:
        return TaxResult(
            taxable_amount=Decimal(str(taxable_amount)),
            applicable_rate=Decimal(str(declared_rate))
            if declared_rate is not None
            else Decimal("0"),
            expected_tax=Decimal("0"),
            declared_tax=(
                Decimal(str(declared_tax))
                if declared_tax is not None
                else None
            ),
            difference=None,
            is_valid=False,
            explanation=rate_result["reason"],
        )

    rate = rate_result["rate"]

    expected_tax = calculate_tax(
        taxable_amount,
        rate,
    )

    declared = Decimal(str(declared_tax))

    difference = calculate_difference(
        declared,
        expected_tax,
    )

    arithmetic_valid = abs(difference) <= tolerance

    rule = rate_result["rule"]

    return TaxResult(
        taxable_amount=Decimal(str(taxable_amount)),
        applicable_rate=rate,
        expected_tax=expected_tax,
        declared_tax=declared,
        difference=difference,
        is_valid=arithmetic_valid,
        rule_id=rule.rule_id if rule else None,
        explanation=(
            "Declared tax matches the calculated tax."
            if arithmetic_valid
            else "Declared tax does not match the calculated tax."
        ),
        source_document=(
            rule.source_document if rule else None
        ),
        source_reference=(
            rule.source_reference if rule else None
        ),
    )
