from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


# ============================================================
# Decimal Helper
# ============================================================

def to_decimal(value) -> Decimal:
    """
    Safely convert a numeric value to Decimal.

    Decimal is used instead of floating-point arithmetic
    because this application deals with monetary values.
    """

    if pd.isna(value):
        raise ValueError(
            "Cannot convert missing value to Decimal."
        )

    return Decimal(str(value))


# ============================================================
# Calculate Sales Tax
# ============================================================

def calculate_sales_tax(
    taxable_amount,
    tax_rate
) -> Decimal:
    """
    Calculate Sales Tax.

    Formula
    -------
    Sales Tax = Taxable Amount × Tax Rate / 100

    Parameters
    ----------
    taxable_amount:
        Value excluding Sales Tax.

    tax_rate:
        Sales Tax rate expressed as a percentage.

    Returns
    -------
    Decimal
        Calculated Sales Tax rounded to 2 decimal places.
    """

    amount = to_decimal(taxable_amount)
    rate = to_decimal(tax_rate)

    tax = (
        amount * rate / Decimal("100")
    )

    return tax.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )


# ============================================================
# Compare Declared vs Expected Tax
# ============================================================

def compare_sales_tax(
    taxable_amount,
    tax_rate,
    declared_tax_amount,
    tolerance=Decimal("0.01")
) -> dict:
    """
    Compare the declared Sales Tax with the
    mathematically expected Sales Tax.

    This function does NOT determine whether the
    tax rate itself is legally correct.

    It only checks the arithmetic.
    """

    expected_tax = calculate_sales_tax(
        taxable_amount,
        tax_rate
    )

    declared_tax = to_decimal(
        declared_tax_amount
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

    difference = (
        declared_tax - expected_tax
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )

    is_valid = (
        abs(difference) <= tolerance
    )

    return {
        "expected_tax": expected_tax,
        "declared_tax": declared_tax,
        "difference": difference,
        "is_valid": is_valid,
    }


# ============================================================
# Validate One Invoice Row
# ============================================================

def validate_invoice_tax(
    row: pd.Series
) -> dict:
    """
    Validate Sales Tax arithmetic for one invoice row.
    """

    result = compare_sales_tax(
        taxable_amount=row[
            "value_excluding_sales_tax"
        ],
        tax_rate=row[
            "sales_tax_rate"
        ],
        declared_tax_amount=row[
            "sales_tax_amount"
        ],
    )

    return result


# ============================================================
# Validate Entire Invoice DataFrame
# ============================================================

def validate_invoice_tax_dataframe(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate Sales Tax calculations for every invoice row.

    Adds the following columns:

        expected_sales_tax
        tax_difference
        tax_calculation_valid
    """

    df = df.copy()

    expected_taxes = []
    differences = []
    validity = []

    for _, row in df.iterrows():

        result = validate_invoice_tax(row)

        expected_taxes.append(
            float(result["expected_tax"])
        )

        differences.append(
            float(result["difference"])
        )

        validity.append(
            result["is_valid"]
        )

    df["expected_sales_tax"] = expected_taxes

    df["tax_difference"] = differences

    df["tax_calculation_valid"] = validity

    return df