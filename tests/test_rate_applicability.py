from src.fbr.rate_applicability import (
    FBRRateApplicabilityResolver,
)


def test_standard_rate_selected_over_further_tax():

    resolver = FBRRateApplicabilityResolver()

    candidates = [
        {
            "rate": 4.0,
            "category": "further",
            "confidence": 0.90,
            "source": "Sales Tax Act 2024",
            "page": 30,
        },
        {
            "rate": 18.0,
            "category": "standard",
            "confidence": 0.88,
            "source": "Sales Tax Act 2025",
            "page": 27,
        },
    ]

    result = resolver.resolve(
        candidates
    )

    assert result.base_rate == 18.0
    assert result.additional_rate == 4.0
    assert result.category == "standard"


def test_standard_rate_selected_from_multiple_standard_rates():

    resolver = FBRRateApplicabilityResolver()

    candidates = [
        {
            "rate": 17.0,
            "category": "standard",
            "confidence": 0.80,
            "year": 2022,
        },
        {
            "rate": 18.0,
            "category": "standard",
            "confidence": 0.90,
            "year": 2025,
        },
    ]

    result = resolver.resolve(
        candidates
    )

    assert result.base_rate == 18.0


def test_reduced_rate():

    resolver = FBRRateApplicabilityResolver()

    candidates = [
        {
            "rate": 5.0,
            "category": "reduced",
            "confidence": 0.91,
            "source": "FBR SRO",
            "page": 1,
        }
    ]

    result = resolver.resolve(
        candidates
    )

    assert result.base_rate == 5.0
    assert result.category == "reduced"


def test_enhanced_rate():

    resolver = FBRRateApplicabilityResolver()

    candidates = [
        {
            "rate": 25.0,
            "category": "enhanced",
            "confidence": 0.92,
            "source": "FBR SRO",
            "page": 1,
        }
    ]

    result = resolver.resolve(
        candidates
    )

    assert result.base_rate == 25.0
    assert result.category == "enhanced"


def test_empty_candidates():

    resolver = FBRRateApplicabilityResolver()

    result = resolver.resolve([])

    assert result.base_rate is None
    assert result.category == "unknown"