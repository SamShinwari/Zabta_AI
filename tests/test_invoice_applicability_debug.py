from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)
from src.fbr.invoice_rate_resolver import (
    FBRInvoiceRateResolver,
)


def test_invoice_applicability_debug():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=10,
    )

    resolver = FBRInvoiceRateResolver(
        current_rate_service=service
    )

    query = resolver.build_query(
        item_description="Taxable goods",
        hs_code="8471.30",
        invoice_date="2026-08-19",
        purchase_type="local purchase",
        invoice_type="taxable",
    )

    results = service.retrieve(query)

    candidates = (
        resolver._build_applicability_candidates(
            results
        )
    )

    print()
    print("=" * 80)
    print("APPLICABILITY CANDIDATES")
    print("=" * 80)

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):
        print()
        print(f"Candidate #{i}")
        print("Rate:", candidate.get("rate"))
        print("Category:", candidate.get("category"))
        print("Applicability:", candidate.get("applicability"))
        print("Year:", candidate.get("year"))
        print(
            "Retrieval:",
            candidate.get("retrieval_score"),
        )
        print(
            "Authority:",
            candidate.get("authority_score"),
        )
        print(
            "Effective from:",
            candidate.get("effective_from"),
        )
        print(
            "Effective to:",
            candidate.get("effective_to"),
        )
        print(
            "Date relevance:",
            candidate.get("date_relevance_score"),
        )
        print("-" * 80)

    print()
    print("=" * 80)

    assert candidates