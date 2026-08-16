from src.fbr.reranker import FBRReranker


def test_reranker_weights():

    reranker = FBRReranker()

    total = (
        reranker.semantic_weight
        + reranker.recency_weight
        + reranker.authority_weight
    )

    assert abs(total - 1.0) < 1e-6


def test_authority_sales_tax_act():

    reranker = FBRReranker()

    score = reranker.authority_score(
        "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    assert score == 1.0


def test_authority_sales_tax_rules():

    reranker = FBRReranker()

    score = reranker.authority_score(
        "The Sales Tax Rules, 2006.pdf"
    )

    assert score == 0.95


def test_extract_year():

    reranker = FBRReranker()

    year = reranker.extract_year(
        "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    assert year == 2026


def test_recency():

    reranker = FBRReranker()

    score_2026 = reranker.recency_score(
        "Sales Tax Act 1990 amended upto 30-06-2026.pdf",
        current_year=2026,
    )

    score_2022 = reranker.recency_score(
        "Finance Act, 2022.pdf",
        current_year=2026,
    )

    assert score_2026 > score_2022


def test_rerank():

    reranker = FBRReranker()

    results = [
        {
            "rank": 1,
            "score": 0.689,
            "text": "Old document",
            "metadata": {
                "source": "Finance Act, 2022.pdf",
                "page": 128,
            },
        },
        {
            "rank": 2,
            "score": 0.683,
            "text": "Current document",
            "metadata": {
                "source": (
                    "Sales Tax Act 1990 "
                    "amended upto 30-06-2026.pdf"
                ),
                "page": 28,
            },
        },
    ]

    reranked = reranker.rerank(
        results,
        current_year=2026,
    )

    assert len(reranked) == 2

    assert (
        reranked[0]["metadata"]["source"]
        == (
            "Sales Tax Act 1990 "
            "amended upto 30-06-2026.pdf"
        )
    )

    assert reranked[0]["rank"] == 1
