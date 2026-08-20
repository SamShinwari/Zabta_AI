from pathlib import Path

import pandas as pd

from src.invoice.invoice_parser import InvoiceParser


def test_invoice_parser():

    parser = InvoiceParser()

    data = {
        "invoice_number": [
            "INV-2026-000001",
        ],
        "invoice_date": [
            "2025-04-15",
        ],
        "seller_name": [
            "Tech World Pakistan",
        ],
        "seller_ntn": [
            "2345678-9",
        ],
        "seller_strn": [
            "3277987654322",
        ],
        "buyer_name": [
            "Quetta Retailer",
        ],
        "buyer_ntn": [
            "1111111-1",
        ],
        "item_description": [
            "Graphics Card",
        ],
        "quantity": [
            9,
        ],
        "unit_price": [
            85000,
        ],
        "taxable_amount": [
            765000,
        ],
        "gst_rate": [
            0.18,
        ],
        "tax_amount": [
            137700,
        ],
    }

    df = pd.DataFrame(data)

    normalized = parser.normalize(df)

    assert len(normalized) == 1

    assert (
        normalized.iloc[0]["item_description"]
        == "Graphics Card"
    )

    assert (
        normalized.iloc[0]["taxable_amount"]
        == 765000
    )

    assert (
        normalized.iloc[0]["gst_rate"]
        == 0.18
    )


def test_invoice_validation():

    parser = InvoiceParser()

    row = pd.Series(
        {
            "invoice_number": "INV-2026-000001",
            "invoice_date": pd.Timestamp(
                "2025-04-15"
            ),
            "item_description": "Graphics Card",
            "quantity": 9,
            "unit_price": 85000,
            "taxable_amount": 765000,
            "gst_rate": 0.18,
            "tax_amount": 137700,
        }
    )

    result = parser.validate_invoice(row)

    assert result["valid"] is True
    assert result["errors"] == []