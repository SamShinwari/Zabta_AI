from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)


def test_current_standard_rate():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=20,
    )

    result = service.resolve_standard_rate(
        invoice_date="2026-08-19",
        top_k=20,
    )

    print()
    print("=" * 80)
    print("CURRENT STANDARD SALES TAX RATE")
    print("=" * 80)

    print(
        "Rate:",
        result.rate,
    )

    print(
        "Source:",
        result.source,
    )

    print(
        "Page:",
        result.page,
    )

    print(
        "Year:",
        result.year,
    )

    print("=" * 80)

    assert result is not None
    assert result.rate == 18.0