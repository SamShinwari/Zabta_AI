import re

import pandas as pd


# ============================================================
# Column Name Normalization
# ============================================================

def normalize_column_name(column: str) -> str:
    """
    Normalize a single invoice column name.

    Examples
    --------
    "Invoice Number"
        -> "invoice_number"

    "Invoice Date"
        -> "invoice_date"

    "Sales Tax Rate"
        -> "sales_tax_rate"

    "Value Excluding Sales Tax"
        -> "value_excluding_sales_tax"
    """

    column = str(column).strip().lower()

    # Replace every non-alphanumeric character
    # with an underscore.
    column = re.sub(
        r"[^a-z0-9]+",
        "_",
        column
    )

    # Remove underscores from beginning/end.
    column = column.strip("_")

    return column


# ============================================================
# Normalize All DataFrame Columns
# ============================================================

def normalize_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize all column names in an invoice DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw invoice DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized column names.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    return df


# ============================================================
# Clean Text Columns
# ============================================================

def clean_text_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove unnecessary whitespace from text columns.
    """

    df = df.copy()

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


# ============================================================
# Clean Invoice Date
# ============================================================

def clean_invoice_date(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert invoice_date to pandas datetime.

    Invalid dates become NaT and are handled later
    by the validation module.
    """

    df = df.copy()

    if "invoice_date" in df.columns:

        df["invoice_date"] = pd.to_datetime(
            df["invoice_date"],
            errors="coerce"
        )

    return df


# ============================================================
# Clean Numeric Columns
# ============================================================

def clean_numeric_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert Sales Tax related numeric fields
    into numeric pandas values.

    Invalid numeric values become NaN and are
    handled later by validation.
    """

    df = df.copy()

    numeric_columns = [
        "quantity",
        "unit_price",
        "value_excluding_sales_tax",
        "sales_tax_rate",
        "sales_tax_amount",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# Clean HS Code
# ============================================================

def clean_hs_code(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean HS codes without converting them to numeric.

    HS codes should remain strings because:
    - leading zeros may be significant
    - decimal formatting may be used
    - HS codes are identifiers, not quantities
    """

    df = df.copy()

    if "hs_code" in df.columns:

        df["hs_code"] = (
            df["hs_code"]
            .astype("string")
            .str.strip()
        )

    return df


# ============================================================
# Clean Taxpayer Identifiers
# ============================================================

def clean_taxpayer_identifiers(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean supplier and buyer STRN/NTN/CNIC fields.

    These identifiers remain strings because they are
    identifiers rather than numerical quantities.
    """

    df = df.copy()

    identifier_columns = [
        "supplier_strn_ntn_cnic",
        "buyer_strn_ntn_cnic",
    ]

    for column in identifier_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    return df


# ============================================================
# Remove Completely Empty Rows
# ============================================================

def remove_empty_rows(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove rows where every field is empty.
    """

    df = df.copy()

    df = df.dropna(
        how="all"
    )

    return df


# ============================================================
# Remove Duplicate Invoice Rows
# ============================================================

def remove_exact_duplicate_rows(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove completely identical rows.

    Important:
    This only removes exact duplicate rows.

    It does NOT remove invoices having the same
    invoice_number, because the same invoice number
    may appear on multiple rows for different items.
    """

    df = df.copy()

    df = df.drop_duplicates()

    return df


# ============================================================
# Complete Invoice Cleaning Pipeline
# ============================================================

def clean_invoice(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Complete invoice normalization and cleaning pipeline.

    Processing order:

    1. Remove completely empty rows
    2. Normalize column names
    3. Clean text fields
    4. Clean invoice dates
    5. Clean numeric fields
    6. Clean HS codes
    7. Clean taxpayer identifiers
    8. Remove exact duplicate rows

    Parameters
    ----------
    df : pandas.DataFrame
        Raw invoice DataFrame.

    Returns
    -------
    pandas.DataFrame
        Cleaned invoice DataFrame.
    """

    # Work on a copy so the original DataFrame
    # is not modified.
    df = df.copy()

    # --------------------------------------------------------
    # Step 1: Remove completely empty rows
    # --------------------------------------------------------

    df = remove_empty_rows(df)

    # --------------------------------------------------------
    # Step 2: Normalize column names
    # --------------------------------------------------------

    df = normalize_columns(df)

    # --------------------------------------------------------
    # Step 3: Clean text columns
    # --------------------------------------------------------

    df = clean_text_columns(df)

    # --------------------------------------------------------
    # Step 4: Clean invoice date
    # --------------------------------------------------------

    df = clean_invoice_date(df)

    # --------------------------------------------------------
    # Step 5: Clean numeric fields
    # --------------------------------------------------------

    df = clean_numeric_columns(df)

    # --------------------------------------------------------
    # Step 6: Clean HS codes
    # --------------------------------------------------------

    df = clean_hs_code(df)

    # --------------------------------------------------------
    # Step 7: Clean taxpayer identifiers
    # --------------------------------------------------------

    df = clean_taxpayer_identifiers(df)

    # --------------------------------------------------------
    # Step 8: Remove exact duplicate rows
    # --------------------------------------------------------

    df = remove_exact_duplicate_rows(df)

    return df