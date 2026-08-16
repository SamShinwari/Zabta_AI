from pathlib import Path

from src.fbr.retriever import FBRRetriever


VECTOR_DIR = Path(
    "data/vector_database/fbr"
)


def test_vector_store_loads():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    assert retriever.vector_count == 153157
    assert retriever.dimension == 1024


def test_sales_tax_rate_retrieval():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    results = retriever.search(
        "What is the standard rate of sales tax under the Sales Tax Act 1990?",
        top_k=5,
    )

    assert len(results) == 5

    assert any(
        "Sales Tax Act" in result["metadata"].get(
            "source",
            "",
        )
        for result in results
    )


def test_registration_retrieval():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    results = retriever.search(
        "Who is required to register for sales tax?",
        top_k=5,
    )

    assert len(results) == 5


def test_return_retrieval():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    results = retriever.search(
        "What is the procedure for filing a sales tax return?",
        top_k=5,
    )

    assert len(results) == 5


def test_input_output_tax_retrieval():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    results = retriever.search(
        "What are input tax and output tax?",
        top_k=5,
    )

    assert len(results) == 5


def test_penalty_retrieval():

    retriever = FBRRetriever(
        VECTOR_DIR
    )

    results = retriever.search(
        "What are the penalties for non-compliance with sales tax requirements?",
        top_k=5,
    )

    assert len(results) == 5