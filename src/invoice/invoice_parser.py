from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class InvoiceParser:
    """
    Load and normalize invoice data from CSV/TSV files.

    The parser does not perform FBR tax-rate resolution.
    It only prepares invoice information for the existing
    Zabta tax-resolution pipeline.
    """

    REQUIRED_COLUMNS = [
        "invoice_number",
        "invoice_date",
        "seller_name",
        "seller_ntn",
        "seller_strn",
        "buyer_name",
        "buyer_ntn",
        "item_description",
        "quantity",
        "unit_price",
        "taxable_amount",
        "gst_rate",
        "tax_amount",
    ]

    NUMERIC_COLUMNS = [
        "quantity",
        "unit_price",
        "taxable_amount",
        "gst_rate",
        "tax_amount",
    ]

    STRING_COLUMNS = [
        "invoice_number",
        "seller_name",
        "seller_ntn",
        "seller_strn",
        "buyer_name",
        "buyer_ntn",
        "item_description",
    ]

    def __init__(self) -> None:
        pass

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        file_path: str | Path,
    ) -> pd.DataFrame:
        """
        Load an invoice CSV/TSV file.

        Supported:
            .csv
            .tsv
            .txt
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Invoice file not found: {path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)

        elif suffix in (".tsv", ".txt"):
            df = pd.read_csv(
                path,
                sep="\t",
            )

        else:
            raise ValueError(
                "Unsupported invoice format. "
                "Use CSV or TSV."
            )

        return self.normalize(df)

    # ========================================================
    # NORMALIZE
    # ========================================================

    def normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize invoice columns and data types.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        df = df.copy()

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        df.columns = [
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            for column in df.columns
        ]

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        self.validate_columns(df)

        # ----------------------------------------------------
        # String fields
        # ----------------------------------------------------

        for column in self.STRING_COLUMNS:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # ----------------------------------------------------
        # Invoice date
        # ----------------------------------------------------

        df["invoice_date"] = pd.to_datetime(
            df["invoice_date"],
            errors="coerce",
        )

        # ----------------------------------------------------
        # Numeric fields
        # ----------------------------------------------------

        for column in self.NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_columns(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Ensure all required invoice columns exist.
        """

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required invoice columns: "
                + ", ".join(missing)
            )

    # ========================================================
    # REQUIRED FIELD VALIDATION
    # ========================================================

    def validate_invoice(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:
        """
        Validate one invoice row.

        Returns a structured validation report.
        """

        errors: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------
        # Invoice number
        # ----------------------------------------------------

        if not str(
            row.get("invoice_number", "")
        ).strip():

            errors.append(
                "Invoice number is missing."
            )

        # ----------------------------------------------------
        # Invoice date
        # ----------------------------------------------------

        if pd.isna(
            row.get("invoice_date")
        ):

            errors.append(
                "Invoice date is missing or invalid."
            )

        # ----------------------------------------------------
        # Item
        # ----------------------------------------------------

        if not str(
            row.get("item_description", "")
        ).strip():

            errors.append(
                "Item description is missing."
            )

        # ----------------------------------------------------
        # Quantity
        # ----------------------------------------------------

        quantity = row.get("quantity")

        if pd.isna(quantity):

            errors.append(
                "Quantity is missing."
            )

        elif quantity <= 0:

            errors.append(
                "Quantity must be greater than zero."
            )

        # ----------------------------------------------------
        # Unit price
        # ----------------------------------------------------

        unit_price = row.get("unit_price")

        if pd.isna(unit_price):

            errors.append(
                "Unit price is missing."
            )

        elif unit_price < 0:

            errors.append(
                "Unit price cannot be negative."
            )

        # ----------------------------------------------------
        # Taxable amount
        # ----------------------------------------------------

        taxable_amount = row.get(
            "taxable_amount"
        )

        if pd.isna(taxable_amount):

            errors.append(
                "Taxable amount is missing."
            )

        elif taxable_amount < 0:

            errors.append(
                "Taxable amount cannot be negative."
            )

        # ----------------------------------------------------
        # GST rate
        # ----------------------------------------------------

        gst_rate = row.get("gst_rate")

        if pd.isna(gst_rate):

            warnings.append(
                "GST rate is missing."
            )

        # ----------------------------------------------------
        # Tax amount
        # ----------------------------------------------------

        tax_amount = row.get("tax_amount")

        if pd.isna(tax_amount):

            warnings.append(
                "Tax amount is missing."
            )

        # ----------------------------------------------------
        # Return report
        # ----------------------------------------------------

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ========================================================
    # CONVERT ROW
    # ========================================================

    def row_to_dict(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:
        """
        Convert one normalized invoice row into
        a JSON-friendly dictionary.
        """

        result = row.to_dict()

        if pd.notna(
            result.get("invoice_date")
        ):

            result["invoice_date"] = (
                result["invoice_date"]
                .strftime("%Y-%m-%d")
            )

        return result