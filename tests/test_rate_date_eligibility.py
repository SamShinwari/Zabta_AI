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
            f"Sales Tax Act amended upto "
            f"{effective_from}"
        ),
        page=30,
        text=(
            "There shall be charged, levied "
            "and paid a tax known as sales tax "
            "at the rate of eighteen per cent "
            "of the value of taxable supplies."
        ),
        authority_score=1.0,
        semantic_score=0.90,
        retrieval_score=retrieval_score,
        year=year,
        effective_from=effective_from,
    )


def test_newer_document_is_preferred_for_invoice_date():

    resolver = FBRRateResolver()

    candidates = [
        make_candidate(
            rate=17.0,
            effective_from="2022-08-22",
            year=2022,
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


def test_future_document_is_not_preferred():

    resolver = FBRRateResolver()

    candidates = [
        make_candidate(
            rate=18.0,
            effective_from="2025-06-30",
            year=2025,
        ),
        make_candidate(
            rate=20.0,
            effective_from="2027-06-30",
            year=2027,
        ),
    ]

    ranked = resolver.rank_candidates(
        candidates,
        invoice_date="2026-08-19",
    )

    assert ranked[0].rate == 18.0


def test_old_document_can_still_be_used_when_no_newer_document_exists():

    resolver = FBRRateResolver()

    candidates = [
        make_candidate(
            rate=17.0,
            effective_from="2022-08-22",
            year=2022,
        ),
    ]

    ranked = resolver.rank_candidates(
        candidates,
        invoice_date="2026-08-19",
    )

    assert ranked
    assert ranked[0].rate == 17.0