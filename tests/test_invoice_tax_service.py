from decimal import Decimal

from src.fbr.invoice_tax_service import (
    ZabtaInvoiceTaxService,
)


def test_invoice_tax_service():

    service = ZabtaInvoiceTaxService(
        retrieval_top_k=10,
    )

    result = service.calculate(
        taxable_amount=100000,
        question=(
            "What is the standard sales tax rate "
            "applicable to taxable supplies in Pakistan?"
        ),
    )

    print()
    print("=" * 80)
    print("ZABTA INVOICE TAX CALCULATION")
    print("=" * 80)

    print(
        "Taxable amount:",
        result.taxable_amount,
    )

    print(
        "Applicable rate:",
        result.applicable_rate,
    )

    print(
        "Rate category:",
        result.rate_category,
    )

    print(
        "Sales tax:",
        result.sales_tax_amount,
    )

    print(
        "Source:",
        result.source_document,
    )

    print(
        "Page:",
        result.source_page,
    )

    print(
        "Confidence:",
        result.confidence,
    )

    assert result.taxable_amount == Decimal(
        "100000"
    )

    assert result.applicable_rate > 0

    assert result.sales_tax_amount > 0

    assert result.source_document

    assert result.confidence > 0