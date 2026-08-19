from src.fbr.rate_resolver import (
    FBRRateResolver,
    TaxRateCandidate,
)


def make_candidate(
    rate,
    effective_from,
    year,
    retrieval_score=0.90,
):
    return TaxRateCandidate(
        rate=rate,
        source=(
            f"Sales Tax Act, 1990 "
            f"amended upto {effective_from}"
        ),
        page=30,
        text=(
            "Scope of tax. There shall be charged, "
            "levied and paid a tax known as sales tax "
            f"at the rate of {rate} per cent of the "
            "value of taxable supplies."
        ),
        authority_score=1.0,
        semantic_score=0.90,
        retrieval_score=retrieval_score,
        year=year,
        effective_from=effective_from,
    )


def test_latest_applicable_rate_wins_for_2026_invoice():

    resolver = FBRRateResolver()

    candidates = [
        make_candidate(
            rate=17.0,
            effective_from="2022-08-22",
            year=2022,
        ),
        make_candidate(
            rate=18.0,
            effective_from="2023-06-30",
            year=2023,
        ),
        make_candidate(
            rate=18.0,
            effective_from="2024-06-30",
            year=2024,
        ),
        make_candidate(
            rate=18.0,
            effective_from="2025-06-30",
            year=2025,
        ),
    ]

    ranked = resolver.rank_candidates(
        candidates,
        invoice_date="2026-08-19",
    )

    assert ranked[0].rate == 18.0
    assert ranked[0].effective_from == "2025-06-30"


def test_2022_rate_does_not_beat_2025_rate():

    resolver = FBRRateResolver()

    candidates = [
        make_candidate(
            rate=17.0,
            effective_from="2022-08-22",
            year=2022,
            retrieval_score=0.99,
        ),
        make_candidate(
            rate=18.0,
            effective_from="2025-06-30",
            year=2025,
            retrieval_score=0.70,
        ),
    ]

    ranked = resolver.rank_candidates(
        candidates,
        invoice_date="2026-08-19",
    )

    assert ranked[0].rate == 18.0