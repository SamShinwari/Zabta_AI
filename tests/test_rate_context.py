from src.fbr.rate_resolver import FBRRateResolver


def test_standard_rate_context():

    resolver = FBRRateResolver()

    category, applicability = (
        resolver.classify_rate_context(
            rate=18.0,
            text=(
                "there shall be charged, levied and paid "
                "a tax known as sales tax at the rate of "
                "[eighteen] per cent of the value"
            ),
        )
    )

    assert category == "standard"
    assert applicability == "general"


def test_further_tax_context():

    resolver = FBRRateResolver()

    category, applicability = (
        resolver.classify_rate_context(
            rate=4.0,
            text=(
                "there shall be charged, levied and paid "
                "a further tax at the rate of [four] "
                "percent of the value"
            ),
        )
    )

    assert category == "further"
    assert applicability == "conditional"


def test_enhanced_rate_context():

    resolver = FBRRateResolver()

    category, applicability = (
        resolver.classify_rate_context(
            rate=25.0,
            text=(
                "enhanced rate of 25% sales tax "
                "shall apply to specified goods"
            ),
        )
    )

    assert category == "enhanced"


def test_reduced_rate_context():

    resolver = FBRRateResolver()

    category, applicability = (
        resolver.classify_rate_context(
            rate=5.0,
            text=(
                "a reduced rate of 5% sales tax "
                "shall apply to specified goods"
            ),
        )
    )

    assert category == "reduced"