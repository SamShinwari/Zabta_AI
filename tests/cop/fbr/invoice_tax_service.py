from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)
from src.tax_engine.tax_calculator import (
    calculate_tax,
)


@dataclass
class InvoiceTaxResult:
    """
    Final invoice sales-tax calculation produced by Zabta.

    The tax rate comes from the FBR retrieval/rate-resolution
    layer rather than being permanently hard-coded.
    """

    taxable_amount: Decimal
    applicable_rate: Decimal
    sales_tax_amount: Decimal
    rate_category: str

    source_document: str
    source_page: int | None
    source_chunk: int | None

    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxable_amount": str(
                self.taxable_amount
            ),
            "applicable_rate": str(
                self.applicable_rate
            ),
            "sales_tax_amount": str(
                self.sales_tax_amount
            ),
            "rate_category": self.rate_category,
            "source_document": self.source_document,
            "source_page": self.source_page,
            "source_chunk": self.source_chunk,
            "confidence": self.confidence,
        }


class ZabtaInvoiceTaxService:
    """
    Connect FBR RAG rate resolution with the
    existing Zabta tax calculation engine.

    Flow:

        Invoice information
              ↓
        FBRCurrentRateService
              ↓
        Current FBR rate
              ↓
        tax_engine.calculate_tax()
              ↓
        InvoiceTaxResult
    """

    def __init__(
        self,
        vector_dir: str = "data/vector_database/fbr",
        retrieval_top_k: int = 10,
    ):
        self.rate_service = FBRCurrentRateService(
            vector_dir=vector_dir,
            retrieval_top_k=retrieval_top_k,
        )

    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(
        self,
        taxable_amount,
        question: str,
    ) -> InvoiceTaxResult:
        """
        Resolve the applicable FBR rate and calculate
        sales tax.

        Parameters
        ----------
        taxable_amount:
            Invoice taxable value.

        question:
            FBR query describing the invoice/item for
            which the applicable rate is required.
        """

        amount = Decimal(
            str(taxable_amount)
        )

        if amount < 0:
            raise ValueError(
                "taxable_amount cannot be negative"
            )

        if not question.strip():
            raise ValueError(
                "question cannot be empty"
            )

        # ----------------------------------------------------
        # 1. Resolve current FBR rate
        # ----------------------------------------------------

        rate_result = (
            self.rate_service.resolve(
                question
            )
        )

        # ----------------------------------------------------
        # 2. Convert rate
        # ----------------------------------------------------

        rate = Decimal(
            str(rate_result.rate)
        )

        # ----------------------------------------------------
        # 3. Calculate sales tax
        # ----------------------------------------------------

        sales_tax = calculate_tax(
            taxable_amount=amount,
            tax_rate=rate,
        )

        # ----------------------------------------------------
        # 4. Return complete result
        # ----------------------------------------------------

        return InvoiceTaxResult(
            taxable_amount=amount,
            applicable_rate=rate,
            sales_tax_amount=sales_tax,
            rate_category=(
                rate_result.category
            ),
            source_document=(
                rate_result.source_document
            ),
            source_page=(
                rate_result.page
            ),
            source_chunk=(
                rate_result.chunk
            ),
            confidence=(
                rate_result.confidence
            ),
        )