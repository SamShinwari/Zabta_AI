from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.fbr.invoice_rate_resolver import (
    FBRInvoiceRateResolver,
)


@dataclass
class InvoiceValidationResult:
    """
    Final validation result for one invoice item.
    """

    invoice_number: str
    invoice_date: str
    item_description: str

    declared_rate: float | None
    applicable_rate: float | None

    taxable_amount: float
    declared_tax: float
    expected_tax: float

    rate_match: bool
    tax_match: bool

    category: str
    confidence: float

    source: str
    page: Any

    status: str
    explanation: str


class InvoiceValidationService:
    """
    Connect invoice information with the existing
    FBR invoice-rate resolver and deterministic
    tax calculation.

    Pipeline:

        Invoice
            ↓
        FBR Invoice Rate Resolver
            ↓
        Applicable FBR Rate
            ↓
        Expected Tax
            ↓
        Compare Declared Tax
            ↓
        Validation Result
    """

    def __init__(
        self,
        current_rate_service,
    ):
        if current_rate_service is None:
            raise ValueError(
                "current_rate_service cannot be None"
            )

        self.current_rate_service = (
            current_rate_service
        )

        # IMPORTANT:
        # The invoice-specific resolver is separate
        # from FBRCurrentRateService.
        self.invoice_rate_resolver = (
            FBRInvoiceRateResolver(
                current_rate_service=(
                    current_rate_service
                )
            )
        )

    # ========================================================
    # VALIDATE ONE INVOICE
    # ========================================================

    def validate(
        self,
        invoice: dict[str, Any],
    ) -> InvoiceValidationResult:
        """
        Validate one normalized invoice record.

        The invoice contains:

            invoice_number
            invoice_date
            item_description
            taxable_amount
            gst_rate
            tax_amount

        Optional:

            hs_code
            purchase_type
            invoice_type
        """

        # ====================================================
        # INVOICE INFORMATION
        # ====================================================

        invoice_number = str(
            invoice.get(
                "invoice_number",
                "",
            )
        ).strip()

        invoice_date = str(
            invoice.get(
                "invoice_date",
                "",
            )
        ).strip()

        item_description = str(
            invoice.get(
                "item_description",
                "",
            )
        ).strip()

        hs_code = invoice.get(
            "hs_code",
            None,
        )

        purchase_type = invoice.get(
            "purchase_type",
            "local purchase",
        )

        invoice_type = invoice.get(
            "invoice_type",
            "taxable",
        )

        # ====================================================
        # FINANCIAL VALUES
        # ====================================================

        taxable_amount = float(
            invoice.get(
                "taxable_amount",
                0.0,
            )
        )

        declared_tax = float(
            invoice.get(
                "tax_amount",
                0.0,
            )
        )

        # ====================================================
        # DECLARED GST RATE
        # ====================================================

        raw_gst_rate = invoice.get(
            "gst_rate",
            None,
        )

        declared_rate = None

        if raw_gst_rate is not None:

            try:

                declared_rate = float(
                    raw_gst_rate
                )

                # CSV stores:
                #
                #     0.18
                #
                # Internal representation:
                #
                #     18.0

                if (
                    0 < declared_rate <= 1
                ):
                    declared_rate *= 100.0

            except (
                TypeError,
                ValueError,
            ):

                declared_rate = None

        # ====================================================
        # RESOLVE FBR APPLICABLE RATE
        # ====================================================

        resolver_result = (
            self.invoice_rate_resolver.resolve(
                item_description=(
                    item_description
                ),
                hs_code=hs_code,
                invoice_date=invoice_date,
                purchase_type=(
                    purchase_type
                ),
                invoice_type=(
                    invoice_type
                ),
            )
        )

        # ====================================================
        # EXTRACT FBR RESULT
        # ====================================================

        applicable_rate = float(
            resolver_result.rate
        )

        category = str(
            resolver_result.category
        )

        confidence = float(
            getattr(
                resolver_result,
                "confidence",
                0.0,
            )
        )

        source = str(
            getattr(
                resolver_result,
                "source",
                "",
            )
        )

        page = getattr(
            resolver_result,
            "page",
            None,
        )

        # ====================================================
        # EXPECTED TAX
        # ====================================================

        expected_tax = (
            taxable_amount
            * applicable_rate
            / 100.0
        )

        # ====================================================
        # RATE COMPARISON
        # ====================================================

        rate_match = False

        if declared_rate is not None:

            rate_match = (
                abs(
                    declared_rate
                    - applicable_rate
                )
                <= 0.01
            )

        # ====================================================
        # TAX COMPARISON
        # ====================================================

        tax_match = (
            abs(
                declared_tax
                - expected_tax
            )
            <= 0.01
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if (
            rate_match
            and tax_match
        ):

            status = "VALID"

            explanation = (
                "The declared invoice sales-tax "
                "rate and tax amount match the "
                "FBR-resolved applicable rate."
            )

        elif not rate_match:

            status = "RATE MISMATCH"

            explanation = (
                "The sales-tax rate declared on "
                "the invoice does not match the "
                "FBR-resolved applicable rate."
            )

        else:

            status = "TAX MISMATCH"

            explanation = (
                "The invoice tax amount does not "
                "match the tax calculated using "
                "the FBR-resolved applicable rate."
            )

        # ====================================================
        # RESULT
        # ====================================================

        return InvoiceValidationResult(
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            item_description=item_description,
            declared_rate=declared_rate,
            applicable_rate=applicable_rate,
            taxable_amount=taxable_amount,
            declared_tax=declared_tax,
            expected_tax=expected_tax,
            rate_match=rate_match,
            tax_match=tax_match,
            category=category,
            confidence=confidence,
            source=source,
            page=page,
            status=status,
            explanation=explanation,
        )

    # ========================================================
    # VALIDATE DATAFRAME
    # ========================================================

    def validate_dataframe(
        self,
        df,
    ):
        """
        Validate every invoice row in a pandas DataFrame.

        Returns a DataFrame containing the validation results.
        """

        import pandas as pd

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        results = []

        for _, row in df.iterrows():

            invoice = row.to_dict()

            result = self.validate(
                invoice
            )

            results.append(
                {
                    "invoice_number": (
                        result.invoice_number
                    ),
                    "invoice_date": (
                        result.invoice_date
                    ),
                    "item_description": (
                        result.item_description
                    ),
                    "declared_rate": (
                        result.declared_rate
                    ),
                    "applicable_rate": (
                        result.applicable_rate
                    ),
                    "taxable_amount": (
                        result.taxable_amount
                    ),
                    "declared_tax": (
                        result.declared_tax
                    ),
                    "expected_tax": (
                        result.expected_tax
                    ),
                    "rate_match": (
                        result.rate_match
                    ),
                    "tax_match": (
                        result.tax_match
                    ),
                    "category": (
                        result.category
                    ),
                    "confidence": (
                        result.confidence
                    ),
                    "source": (
                        result.source
                    ),
                    "page": (
                        result.page
                    ),
                    "status": (
                        result.status
                    ),
                    "explanation": (
                        result.explanation
                    ),
                }
            )

        return pd.DataFrame(
            results
        )