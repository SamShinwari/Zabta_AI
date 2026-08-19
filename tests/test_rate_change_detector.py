from src.fbr.rate_change_detector import (
    FBRRateChangeDetector,
)


def test_detect_enhanced_rate():

    detector = FBRRateChangeDetector()

    text = """
    Enhanced rate of 25% sales tax
    shall apply to specified luxury goods.
    """

    assert detector.contains_change_language(
        text
    )

    rates = detector.extract_rates(
        text
    )

    assert 25.0 in rates

    assert (
        detector.classify_change(text)
        == "enhanced"
    )


def test_detect_reduced_rate():

    detector = FBRRateChangeDetector()

    text = """
    Sales tax reduced to 5% for
    specified imported goods.
    """

    assert detector.contains_change_language(
        text
    )

    rates = detector.extract_rates(
        text
    )

    assert 5.0 in rates

    assert (
        detector.classify_change(text)
        == "reduced"
    )


def test_detect_zero_rated():

    detector = FBRRateChangeDetector()

    text = """
    The following supplies shall be
    zero-rated under the Sales Tax Act.
    """

    assert detector.contains_change_language(
        text
    )

    assert (
        detector.classify_change(text)
        == "zero_rated"
    )


def test_detect_exemption():

    detector = FBRRateChangeDetector()

    text = """
    The goods specified below shall be
    exempt from sales tax.
    """

    assert detector.contains_change_language(
        text
    )

    assert (
        detector.classify_change(text)
        == "exempt"
    )


def test_extract_change_candidate():

    detector = FBRRateChangeDetector()

    results = [
        {
            "score": 0.80,
            "authority_score": 1.0,
            "retrieval_score": 0.80,
            "text": (
                "S.R.O. 297(I)/2023 "
                "enhanced rate of 25% sales tax "
                "on specified luxury goods."
            ),
            "metadata": {
                "source": (
                    "297_I_2023 "
                    "- Sales Tax enhanced rate.pdf"
                ),
                "page": 1,
            },
        }
    ]

    candidates = detector.detect(
        results
    )

    assert len(candidates) == 1

    assert candidates[0].rate == 25.0

    assert candidates[0].category == "enhanced"

    assert candidates[0].change_detected is True
def test_detect_sales_tax_reduced_to():

    detector = FBRRateChangeDetector()

    text = """
    Sales tax reduced to 5% for
    specified imported goods.
    """

    assert detector.contains_change_language(
        text
    )

    rates = detector.extract_rates(
        text
    )

    assert 5.0 in rates

    assert (
        detector.classify_change(text)
        == "reduced"
    )