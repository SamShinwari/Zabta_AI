from src.fbr.retriever import FBRRetriever


def test_sales_tax_act_has_high_authority():

    score = FBRRetriever._authority_score(
        "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    assert score == 1.00


def test_finance_act_has_high_authority():

    score = FBRRetriever._authority_score(
        "Finance Act 2026.pdf"
    )

    assert score == 0.95


def test_sro_has_high_authority():

    score = FBRRetriever._authority_score(
        "SRO 1234(I)/2026.pdf"
    )

    assert score == 0.90


def test_tax_expenditure_report_has_lower_authority():

    score = FBRRetriever._authority_score(
        "Tax Expenditure Report 2026.pdf"
    )

    assert score == 0.50


def test_unknown_document_has_low_authority():

    score = FBRRetriever._authority_score(
        "random_document.pdf"
    )

    assert score == 0.40