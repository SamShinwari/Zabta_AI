from src.fbr.current_rate_service import FBRCurrentRateService
from src.fbr.invoice_rate_resolver import (
    FBRInvoiceRateResolver,
)


def test_real_invoice_rate_candidates():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=30,
    )

    resolver = FBRInvoiceRateResolver(
        current_rate_service=service,
    )

    query = resolver.build_query(
        item_description="Taxable goods",
        hs_code="8471.30",
        invoice_date="2026-08-19",
        purchase_type="local purchase",
        invoice_type="taxable",
    )

    results = service.retrieve(query)

    print()
    print("=" * 100)
    print("REAL INVOICE RATE CANDIDATES")
    print("=" * 100)
    print("QUERY:")
    print(query)

    candidates = resolver._build_applicability_candidates(
        results
    )

    print()
    print("=" * 100)

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\nCandidate #{number}"
        )

        print(
            f"Rate:              {candidate['rate']}"
        )

        print(
            f"Category:           {candidate['category']}"
        )

        print(
            f"Year:               {candidate['year']}"
        )

        print(
            f"Source:             {candidate['source']}"
        )

        print(
            f"Page:               {candidate['page']}"
        )

        print(
            f"Authority:          "
            f"{candidate['authority_score']}"
        )

        print(
            f"Semantic:           "
            f"{candidate['semantic_score']}"
        )

        print(
            f"Retrieval:          "
            f"{candidate['retrieval_score']}"
        )

        print(
            f"Text preview:\n"
            f"{candidate['text'][:700]}"
        )

        print("-" * 100)

    assert candidates