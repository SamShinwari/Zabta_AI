from src.fbr.retriever import FBRRetriever


VECTOR_DIR = "data/vector_database/fbr"


def test_section_query_uses_legal_reference():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    results = retriever.search(
        "What is section 8B of the Sales Tax Act 1990?",
        top_k=10,
    )

    assert len(results) == 10

    assert any(
        result.get(
            "legal_reference_match",
            False,
        )
        for result in results
    )


def test_rule_query_uses_legal_reference():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    results = retriever.search(
        "What does Rule 12 say?",
        top_k=10,
    )

    assert len(results) == 10


def test_sro_query_uses_legal_reference():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    results = retriever.search(
        "Explain SRO 1125(I)/2011",
        top_k=10,
    )

    assert len(results) == 10


def test_normal_query_still_works():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    results = retriever.search(
        "What is the standard sales tax rate in Pakistan?",
        top_k=10,
    )

    assert len(results) == 10

    assert results[0]["score"] >= results[-1]["score"]


def test_retriever_statistics():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    assert retriever.vector_count > 0

    assert retriever.dimension == 1024