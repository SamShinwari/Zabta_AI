from src.fbr.invoice_rate_query import (
    InvoiceRateQuery,
)


def test_invoice_rate_query():

    query = InvoiceRateQuery(
        item_description="Laptop computer",
        hs_code="8471.30",
        invoice_date="2026-08-19",
        purchase_type="local purchase",
        invoice_type="taxable",
    )

    text = query.build_query()

    print()
    print("=" * 80)
    print("INVOICE RATE QUERY")
    print("=" * 80)
    print(text)

    assert "Laptop computer" in text
    assert "8471.30" in text
    assert "2026-08-19" in text
    assert "local purchase" in text
    assert "taxable" in text
    assert "sales tax rate" in text