from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)
from src.fbr.invoice_rate_query import (
    InvoiceRateQuery,
)


def test_invoice_rate_candidates():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=10,
    )

    query = InvoiceRateQuery(
        item_description="Taxable goods",
        hs_code="8471.30",
        invoice_date="2026-08-19",
        purchase_type="local purchase",
        invoice_type="taxable",
    )

    question = query.build_query()

    results = service.retriever.search(
        question,
        top_k=10,
    )

    candidates = (
        service.rate_resolver.extract_candidates(
            results
        )
    )

    ranked = (
        service.rate_resolver.rank_candidates(
            candidates
        )
    )

    print()
    print("=" * 80)
    print("INVOICE RATE CANDIDATES")
    print("=" * 80)

    for number, candidate in enumerate(
        ranked,
        start=1,
    ):
        print()
        print(f"Candidate #{number}")
        print(f"Rate:       {candidate.rate}")
        print(f"Year:       {candidate.year}")
        print(f"Source:     {candidate.source}")
        print(f"Page:       {candidate.page}")
        print(
            f"Authority:  {candidate.authority_score}"
        )
        print(
            f"Semantic:   {candidate.semantic_score}"
        )
        print(
            f"Retrieval:  {candidate.retrieval_score}"
        )
        print(
            f"Category:   {candidate.category}"
        )

        print(
            f"Applicability: {candidate.applicability}"
        )
        print("-" * 80)
        print(candidate.text[:1000])

    assert candidates