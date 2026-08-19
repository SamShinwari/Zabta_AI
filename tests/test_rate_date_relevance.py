from src.fbr.rate_resolver import (
    FBRRateResolver,
    TaxRateCandidate,
)


def make_candidate(
    rate,
    cutoff,
):
    return TaxRateCandidate(
        rate=rate,
        source="Sales Tax Act.pdf",
        page=30,
        text="sales tax rate",
        authority_score=1.0,
        semantic_score=0.8,
        retrieval_score=0.8,
        year=2025,
        effective_from=cutoff,
    )


def test_future_or_current_cutoff_is_highly_relevant():

    resolver = FBRRateResolver()

    candidate = make_candidate(
        18.0,
        "2026-06-30",
    )

    score = resolver.calculate_date_relevance(
        candidate,
        "2026-08-19",
    )

    assert score > 0.9


def test_old_document_gets_lower_relevance():

    resolver = FBRRateResolver()

    candidate = make_candidate(
        18.0,
        "2023-06-30",
    )

    score = resolver.calculate_date_relevance(
        candidate,
        "2026-08-19",
    )

    assert score < 0.5


def test_missing_cutoff_is_neutral():

    resolver = FBRRateResolver()

    candidate = make_candidate(
        18.0,
        None,
    )

    score = resolver.calculate_date_relevance(
        candidate,
        "2026-08-19",
    )

    assert score == 0.5