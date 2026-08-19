from src.fbr.rate_resolver import FBRRateResolver


def test_extract_cutoff_date():

    resolver = FBRRateResolver()

    result = resolver.extract_cutoff_date(
        "Sales Tax Act 1990 amended upto 30-06-2025.pdf"
    )

    assert result == "2025-06-30"


def test_extract_cutoff_date_dotted():

    resolver = FBRRateResolver()

    result = resolver.extract_cutoff_date(
        "Sales Tax Act 1990 amended upto 30.06.2023.pdf"
    )

    assert result == "2023-06-30"


def test_extract_cutoff_date_missing():

    resolver = FBRRateResolver()

    result = resolver.extract_cutoff_date(
        "Some FBR document.pdf"
    )

    assert result is None
def test_extract_cutoff_date_numeric():
    assert (
        FBRRateResolver.extract_cutoff_date(
            "Sales Tax Act amended upto 30-06-2025"
        )
        == "2025-06-30"
    )


def test_extract_cutoff_date_up_to():
    assert (
        FBRRateResolver.extract_cutoff_date(
            "Sales Tax Act amended up to 30-06-2024"
        )
        == "2024-06-30"
    )


def test_extract_cutoff_date_month_name():
    assert (
        FBRRateResolver.extract_cutoff_date(
            "Sales Tax Act amended upto 11th March, 2019"
        )
        == "2019-03-11"
    )


def test_extract_cutoff_date_invalid():
    assert (
        FBRRateResolver.extract_cutoff_date(
            "Sales Tax Act amended upto 31-02-2025"
        )
        is None
    )