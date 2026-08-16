from pathlib import Path

import pandas as pd

from src.invoice.loader import load_invoice
from src.invoice.normalizer import normalize_columns
from src.invoice.validator import (
    validate_required_columns,
    validate_invoice_columns,
    validate_invoice,
    quality_report,
)

from src.invoice.calculator import (
    calculate_sales_tax,
    compare_sales_tax,
    validate_invoice_tax_dataframe,
)
ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_INVOICE = (
    ROOT_DIR
    / "data"
    / "invoices"
    / "test"
    / "sample_invoices.csv"
)


def test_invoice_loader():

    df = load_invoice(TEST_INVOICE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "invoice_number" in df.columns


def test_column_normalization():

    df = pd.DataFrame(
        {
            "Invoice Number": ["INV-001"],
            "Invoice Date": ["2026-08-01"],
            "Seller Name": ["ABC Traders"],
        }
    )

    result = normalize_columns(df)

    assert "invoice_number" in result.columns
    assert "invoice_date" in result.columns
    assert "seller_name" in result.columns


def test_required_columns():

    df = pd.read_csv(TEST_INVOICE)

    missing = validate_required_columns(df)

    assert missing == []


def test_required_columns_validation():

    df = pd.read_csv(TEST_INVOICE)

    validate_invoice_columns(df)


def test_invoice_validation():

    df = pd.read_csv(TEST_INVOICE)

    errors = validate_invoice(df)

    assert errors == []


def test_quality_report():

    df = pd.read_csv(TEST_INVOICE)

    report = quality_report(df)

    assert report["total_rows"] == 5
    assert report["total_columns"] == 13

def test_calculate_sales_tax():

    result = calculate_sales_tax(
        taxable_amount=100000,
        tax_rate=18
    )

    assert result == 18000

def test_calculate_sales_tax_second_case():

    result = calculate_sales_tax(
        taxable_amount=90000,
        tax_rate=18
    )

    assert result == 16200

def test_compare_correct_sales_tax():

    result = compare_sales_tax(
        taxable_amount=100000,
        tax_rate=18,
        declared_tax_amount=18000,
    )

    assert result["expected_tax"] == 18000
    assert result["declared_tax"] == 18000
    assert result["difference"] == 0
    assert result["is_valid"] is True

def test_compare_incorrect_sales_tax():

    result = compare_sales_tax(
        taxable_amount=100000,
        tax_rate=18,
        declared_tax_amount=17000,
    )

    assert result["expected_tax"] == 18000
    assert result["declared_tax"] == 17000
    assert result["difference"] == -1000
    assert result["is_valid"] is False


def test_invoice_tax_dataframe():

    df = pd.read_csv(TEST_INVOICE)

    result = validate_invoice_tax_dataframe(df)

    assert "expected_sales_tax" in result.columns

    assert "tax_difference" in result.columns

    assert "tax_calculation_valid" in result.columns

    assert result["tax_calculation_valid"].all()

