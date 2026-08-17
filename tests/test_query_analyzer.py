from src.fbr.query_analyzer import FBRQueryAnalyzer


def test_section_query():

    analyzer = FBRQueryAnalyzer()

    result = analyzer.analyze(
        "What is section 8B of the Sales Tax Act 1990?"
    )

    assert result.sections == ["8B"]

    assert result.years == [1990]

    assert result.has_legal_reference()


def test_rule_query():

    analyzer = FBRQueryAnalyzer()

    result = analyzer.analyze(
        "What is rule 12 of the Sales Tax Rules?"
    )

    assert result.rules == ["12"]

    assert result.has_legal_reference()


def test_sro_query():

    analyzer = FBRQueryAnalyzer()

    result = analyzer.analyze(
        "Explain SRO 1842(I)/2023"
    )

    assert result.sros == [
        "1842(I)/2023"
    ]

    assert result.years == [2023]

    assert result.has_legal_reference()


def test_normal_question():

    analyzer = FBRQueryAnalyzer()

    result = analyzer.analyze(
        "What is input tax?"
    )

    assert result.sections == []

    assert result.rules == []

    assert result.sros == []

    assert result.years == []

    assert not result.has_legal_reference()