from src.rag.retriever import FBRRetriever


def test_fbr_retrieval():
    retriever = FBRRetriever(top_k=5)

    results = retriever.retrieve(
        "What is the standard sales tax rate in Pakistan?"
    )

    assert len(results) > 0

    for result in results:
        assert "score" in result
        assert "text" in result
        assert "metadata" in result
        assert "citation" in result