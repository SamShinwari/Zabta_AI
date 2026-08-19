from src.fbr.rate_classifier import (
    FBRRateClassifier,
)


def test_standard_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=18.0,
        text=(
            "Sales tax shall be charged "
            "at the standard rate of 18%."
        ),
    )

    assert result.rate == 18.0
    assert result.category == "standard"
    assert result.confidence >= 0.80


def test_special_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=25.0,
        text=(
            "Enhanced rate of 25% sales tax "
            "shall apply to specified luxury goods."
        ),
    )

    assert result.rate == 25.0
    assert result.category == "special"


def test_reduced_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=5.0,
        text=(
            "A reduced rate of 5% sales tax "
            "shall apply to the specified goods."
        ),
    )

    assert result.rate == 5.0
    assert result.category == "reduced"


def test_unknown_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=12.0,
        text=(
            "The applicable rate is 12%."
        ),
    )

    assert result.rate == 12.0
    assert result.category == "unknown"


def test_fbr_section_3_standard_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=18.0,
        text=(
            """
            Section 3.
            Scope of tax.
            There shall be charged, levied
            and paid a tax known as sales tax
            at the rate of eighteen per cent
            of the value of taxable supplies.
            """
        ),
    )

    assert result.category == "standard"


def test_luxury_goods_special_rate():

    classifier = FBRRateClassifier()

    result = classifier.classify(
        rate=25.0,
        text=(
            """
            Enhanced rate of 25% sales tax
            shall be imposed on import and
            supply of luxury goods.
            """
        ),
    )

    assert result.category == "special"