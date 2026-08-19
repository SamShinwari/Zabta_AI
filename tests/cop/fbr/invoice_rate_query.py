from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InvoiceRateQuery:
    """
    Structured information used to resolve the
    applicable FBR sales-tax rate for an invoice item.
    """

    item_description: str
    hs_code: str | None = None
    invoice_date: str | None = None
    purchase_type: str | None = None
    invoice_type: str | None = None

    def build_query(self) -> str:
        """
        Build a precise FBR retrieval query from
        invoice information.
        """

        parts = [
            "Determine the applicable Pakistan sales tax "
            "rate for the following invoice item."
        ]

        if self.item_description:
            parts.append(
                f"Item description: "
                f"{self.item_description}."
            )

        if self.hs_code:
            parts.append(
                f"HS code: {self.hs_code}."
            )

        if self.invoice_date:
            parts.append(
                f"Invoice date: {self.invoice_date}."
            )

        if self.purchase_type:
            parts.append(
                f"Purchase type: {self.purchase_type}."
            )

        if self.invoice_type:
            parts.append(
                f"Invoice type: {self.invoice_type}."
            )

        parts.append(
            "Identify the applicable sales tax rate, "
            "whether it is standard, reduced, enhanced, "
            "zero-rated, exempt, or special, and provide "
            "the supporting FBR document and provision."
        )

        return " ".join(parts)