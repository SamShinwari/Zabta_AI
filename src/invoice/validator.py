import pandas as pd


# ============================================================
# Required and Optional Columns
# ============================================================

REQUIRED_COLUMNS = [
    "invoice_number",
    "invoice_date",
    "supplier_strn_ntn_cnic",
    "buyer_strn_ntn_cnic",
    "invoice_type",
    "purchase_type",
    "item_description",
    "hs_code",
    "quantity",
    "unit_price",
    "value_excluding_sales_tax",
    "sales_tax_rate",
    "sales_tax_amount",
]


OPTIONAL_COLUMNS = [
    "product_category",
    "declared_tax_rate",
    "declared_tax_amount",
]


# ============================================================
# Required Column Validation
# ============================================================

def validate_required_columns(
    df: pd.DataFrame
) -> list[str]:
    """
    Check whether all required invoice columns exist.

    Returns
    -------
    list[str]
        List of missing columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return missing_columns


# ============================================================
# Invoice Column Validation
# ============================================================

def validate_invoice_columns(
    df: pd.DataFrame
) -> None:
    """
    Raise an error if required columns are missing.
    """

    missing_columns = validate_required_columns(df)

    if missing_columns:

        raise ValueError(
            "Missing required invoice columns: "
            + ", ".join(missing_columns)
        )


# ============================================================
# Numeric Validation
# ============================================================

def validate_numeric_columns(
    df: pd.DataFrame
) -> list[str]:

    errors = []

    numeric_columns = [
        "quantity",
        "unit_price",
        "value_excluding_sales_tax",
        "sales_tax_rate",
        "sales_tax_amount",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = converted.isna().sum()

        if invalid_count > 0:

            errors.append(
                f"{column}: "
                f"{invalid_count} invalid values"
            )

    return errors

# ============================================================
# Date Validation
# ============================================================

def validate_invoice_dates(
    df: pd.DataFrame
) -> list[str]:

    errors = []

    if "invoice_date" not in df.columns:
        return errors

    dates = pd.to_datetime(
        df["invoice_date"],
        errors="coerce"
    )

    invalid_count = dates.isna().sum()

    if invalid_count > 0:

        errors.append(
            "invoice_date: "
            f"{invalid_count} invalid dates"
        )

    return errors


# ============================================================
# Complete Invoice Validation
# ============================================================

def validate_invoice(
    df: pd.DataFrame
) -> list[str]:

    errors = []

    # Required columns
    missing_columns = validate_required_columns(df)

    for column in missing_columns:

        errors.append(
            f"Missing column: {column}"
        )

    # Stop here if required columns are missing
    if errors:
        return errors

    # Numeric validation
    errors.extend(
        validate_numeric_columns(df)
    )

    # Date validation
    errors.extend(
        validate_invoice_dates(df)
    )

    return errors


# ============================================================
# Duplicate Invoice Detection
# ============================================================

def find_duplicate_invoices(
    df: pd.DataFrame
) -> pd.DataFrame:

    if "invoice_number" not in df.columns:

        return pd.DataFrame()

    duplicates = df[
        df["invoice_number"].duplicated(
            keep=False
        )
    ]

    return duplicates


# ============================================================
# Invoice Quality Report
# ============================================================

def quality_report(
    df: pd.DataFrame
) -> dict:

    report = {

        "total_rows": len(df),

        "total_columns": len(df.columns),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_invoices": 0,

    }

    if "invoice_number" in df.columns:

        report["duplicate_invoices"] = int(
            df["invoice_number"]
            .duplicated()
            .sum()
        )

    return report
