from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)


def test_standard_rate_retrieval():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=20,
    )

    results = (
        service.retrieve_standard_rate_evidence(
            invoice_date="2026-08-19",
            top_k=20,
        )
    )

    assert results

    print()
    print("=" * 100)
    print("STANDARD RATE RETRIEVAL")
    print("=" * 100)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print(f"Candidate #{index}")

        print(
            "Score:   ",
            result.get("score"),
        )

        print(
            "Source:  ",
            metadata.get("source"),
        )

        print(
            "Page:    ",
            metadata.get("page"),
        )

        print("-" * 100)

        print(
            result.get(
                "text",
                "",
            )[:1000]
        )

    print()
    print("=" * 100)