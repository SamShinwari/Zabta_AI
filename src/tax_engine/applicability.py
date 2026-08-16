from typing import Optional


def determine_tax_applicability(
    invoice_type: Optional[str] = None,
    purchase_type: Optional[str] = None,
    product_category: Optional[str] = None,
) -> dict:
    """
    Determine whether the invoice requires tax-rule resolution.

    This is an MVP decision layer.

    Legal applicability should eventually be resolved from
    the curated FBR rules/RAG knowledge base.
    """

    return {
        "applicable": True,
        "invoice_type": invoice_type,
        "purchase_type": purchase_type,
        "product_category": product_category,
        "reason": (
            "Tax applicability requires rule resolution "
            "from the configured tax rules."
        ),
    }
