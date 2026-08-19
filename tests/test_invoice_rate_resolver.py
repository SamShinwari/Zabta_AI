from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)
from src.fbr.invoice_rate_resolver import (
    FBRInvoiceRateResolver,
)
from src.fbr.rate_resolver import TaxRateCandidate

def test_invoice_rate_resolution():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=10,
    )

    resolver = FBRInvoiceRateResolver(
        current_rate_service=service
    )

    result = resolver.resolve(
        item_description="Taxable goods",
        hs_code="8471.30",
        invoice_date="2026-08-19",
        purchase_type="local purchase",
        invoice_type="taxable",
    )

    print()
    print("=" * 80)
    print("INVOICE RATE RESOLUTION")
    print("=" * 80)

    print(
        f"Item:       {result.item_description}"
    )

    print(
        f"HS Code:    {result.hs_code}"
    )

    print(
        f"Invoice:    {result.invoice_date}"
    )

    print(
        f"Rate:       {result.rate}"
    )

    print(
        f"Category:   {result.category}"
    )

    print(
        f"Confidence: {result.confidence:.4f}"
    )

    print(
        f"Source:     {result.source}"
    )

    print(
        f"Page:       {result.page}"
    )

    assert result.rate is not None
    assert result.rate > 0

    assert result.source is not None

    assert result.confidence > 0
def test_classifies_standard_and_further_rates_separately():

    text = """
    there shall be charged, levied and paid
    a tax known as sales tax at the rate of
    eighteen per cent of the value of taxable supplies.

    where taxable supplies are made to a person
    who has not obtained registration number,
    there shall be charged, levied and paid a
    further tax at the rate of four percent
    of the value.
    """

    standard_candidate = TaxRateCandidate(
        rate=18.0,
        source="Sales Tax Act 2024",
        page=30,
        text=text,
        authority_score=1.0,
        semantic_score=0.8,
        retrieval_score=0.8,
        year=2024,
    )

    further_candidate = TaxRateCandidate(
        rate=4.0,
        source="Sales Tax Act 2024",
        page=30,
        text=text,
        authority_score=1.0,
        semantic_score=0.8,
        retrieval_score=0.8,
        year=2024,
    )

    assert (
        FBRInvoiceRateResolver._classify_rate_candidate(
            standard_candidate
        )
        == "standard"
    )

    assert (
        FBRInvoiceRateResolver._classify_rate_candidate(
            further_candidate
        )
        == "further"
    )
def test_18_percent_not_misclassified_as_zero_rated():

    resolver = FBRInvoiceRateResolver(
        current_rate_service=object()
    )

    candidate = TaxRateCandidate(
        rate=18.0,
        source=(
            "Sales Tax Act 1990 "
            "amended upto 30-06-2026.pdf"
        ),
        page=28,
        text="""
        zero-rated supply means a taxable supply
        which is charged to tax at the rate of
        zero per cent under section 4.

        Chapter-II
        SCOPE AND PAYMENT OF TAX

        3.
        Scope of tax.– Subject to the provisions
        of this Act, there shall be charged,
        levied and paid a tax known as sales tax
        at the rate of eighteen per cent of the
        value of taxable supplies.
        """,
        authority_score=1.0,
        semantic_score=0.9,
        retrieval_score=0.9,
        year=2026,
    )

    category = (
        resolver._classify_candidate(
            candidate
        )
    )

    assert category == "standard"
def test_18_percent_standard_not_classified_as_further_when_same_chunk_contains_further_tax():

    resolver = FBRInvoiceRateResolver(
        current_rate_service=object()
    )

    text = """
    There shall be charged, levied and paid a tax known as
    sales tax at the rate of eighteen per cent of the value of
    taxable supplies made by a registered person.

    Where taxable supplies are made to an unregistered person,
    there shall be charged further tax at the rate of three
    percent of the value.
    """

    standard_candidate = TaxRateCandidate(
        rate=18.0,
        source="Sales Tax Act 1990",
        page=30,
        text=text,
        authority_score=1.0,
        semantic_score=0.9,
        retrieval_score=0.9,
        year=2025,
    )

    further_candidate = TaxRateCandidate(
        rate=3.0,
        source="Sales Tax Act 1990",
        page=30,
        text=text,
        authority_score=1.0,
        semantic_score=0.9,
        retrieval_score=0.9,
        year=2025,
    )

    assert (
        resolver._classify_rate_candidate(
            standard_candidate
        )
        == "standard"
    )

    assert (
        resolver._classify_rate_candidate(
            further_candidate
        )
        == "further"
    )