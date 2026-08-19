from datetime import date

from src.fbr.current_rate_resolver import (
    CurrentRateResolver,
)


def make_result(
    source: str,
    text: str,
    score: float = 0.70,
    rerank_score: float = 0.75,
):
    return {
        "score": score,
        "rerank_score": rerank_score,
        "text": text,
        "metadata": {
            "source": source,
            "page": 30,
        },
    }


def test_document_date_numeric():

    resolver = CurrentRateResolver()

    result = resolver.document_date(
        "Sales Tax Act 1990 amended upto 30-06-2025.pdf"
    )

    assert result == date(
        2025,
        6,
        30,
    )


def test_document_date_text():

    resolver = CurrentRateResolver()

    result = resolver.document_date(
        "Notification dated 8th March, 2023.pdf"
    )

    assert result == date(
        2023,
        3,
        8,
    )


def test_standard_rate_prefers_recent_document():

    resolver = CurrentRateResolver()

    results = [
        make_result(
            source=(
                "Sales Tax Act, 1990 "
                "amended upto 30.06.2023.pdf"
            ),
            text=(
                "Sales tax shall be charged "
                "at the rate of eighteen per cent "
                "of taxable supplies."
            ),
        ),
        make_result(
            source=(
                "Sales Tax Act 1990 "
                "amended upto 30-06-2025.pdf"
            ),
            text=(
                "Sales tax shall be charged "
                "at the standard rate of eighteen "
                "per cent of taxable supplies."
            ),
        ),
    ]

    result = resolver.resolve(
        results,
        as_of=date(
            2026,
            8,
            19,
        ),
    )

    assert result is not None
    assert result.rate == 18.0
    assert result.category == "standard"
    assert "2025" in result.source


def test_special_rate_does_not_replace_standard():

    resolver = CurrentRateResolver()

    results = [
        make_result(
            source=(
                "Sales Tax Act 1990 "
                "amended upto 30-06-2025.pdf"
            ),
            text=(
                "Sales tax shall be charged "
                "at the standard rate of 18%."
            ),
        ),
        make_result(
            source=(
                "SRO 297(I)/2023.pdf"
            ),
            text=(
                "Enhanced rate of 25% sales tax "
                "shall apply to luxury goods."
            ),
            score=0.80,
            rerank_score=0.85,
        ),
    ]

    result = resolver.resolve(
        results,
        as_of=date(
            2026,
            8,
            19,
        ),
    )

    assert result is not None
    assert result.rate == 18.0
    assert result.category == "standard"


def test_can_resolve_special_rate():

    resolver = CurrentRateResolver()

    results = [
        make_result(
            source="SRO 297(I)/2023.pdf",
            text=(
                "Enhanced rate of 25% sales tax "
                "shall apply to luxury goods."
            ),
        ),
    ]

    result = resolver.resolve(
        results,
        as_of=date(
            2026,
            8,
            19,
        ),
        category="special",
    )

    assert result is not None
    assert result.rate == 25.0
    assert result.category == "special"


def test_future_document_is_ignored():

    resolver = CurrentRateResolver()

    results = [
        make_result(
            source=(
                "Sales Tax Act "
                "amended upto 30-06-2027.pdf"
            ),
            text=(
                "Sales tax shall be charged "
                "at the standard rate of 20%."
            ),
        ),
        make_result(
            source=(
                "Sales Tax Act "
                "amended upto 30-06-2025.pdf"
            ),
            text=(
                "Sales tax shall be charged "
                "at the standard rate of 18%."
            ),
        ),
    ]

    result = resolver.resolve(
        results,
        as_of=date(
            2026,
            8,
            19,
        ),
    )

    assert result is not None
    assert result.rate == 18.0