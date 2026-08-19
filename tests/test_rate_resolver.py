from src.fbr.rate_resolver import FBRRateResolver


def test_extract_eighteen_per_cent_without_brackets():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        sales tax at the rate of
        eighteen per cent of the value
        """
    )

    assert 18.0 in rates


def test_extract_numeric_percentage():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        Sales tax shall be charged at 18%
        """
    )

    assert 18.0 in rates


def test_extract_multiple_rates():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        Sales tax is 18%.
        Certain goods may be subject to 25%.
        """
    )

    assert 18.0 in rates
    assert 25.0 in rates


def test_extract_year():

    resolver = FBRRateResolver()

    year = resolver.extract_year(
        "Sales Tax Act 1990 amended upto 30-06-2025.pdf"
    )

    assert year == 2025


def test_extract_eighteen_per_cent():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        there shall be charged, levied
        and paid a tax known as sales tax
        at the rate of [eighteen] per cent
        of the value of taxable supplies.
        """
    )

    assert 18.0 in rates


def test_extract_seventeen_per_cent():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        sales tax at the rate of
        [seventeen] per cent of the value
        of taxable supplies.
        """
    )

    assert 17.0 in rates


def test_extract_twenty_five_per_cent():

    resolver = FBRRateResolver()

    rates = resolver.extract_rates(
        """
        enhanced rate of [twenty-five]
        per cent shall apply to specified goods.
        """
    )

    assert 25.0 in rates