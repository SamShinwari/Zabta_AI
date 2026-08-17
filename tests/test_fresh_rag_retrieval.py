from __future__ import annotations

from src.fbr.retriever import FBRRetriever


VECTOR_DIR = "data/vector_database/fbr"


def print_results(
    query: str,
    results: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("QUERY")
    print("=" * 100)
    print(query)

    print()
    print("=" * 100)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 100)

    for i, result in enumerate(results, start=1):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print("-" * 100)
        print(f"RESULT {i}")
        print("-" * 100)

        print(
            f"Score: {result.get('score'):.4f}"
        )

        print(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        print(
            f"Chunk: {metadata.get('chunk', 'Unknown')}"
        )

        text = result.get(
            "text",
            "",
        ).strip()

        print()
        print("Text:")
        print(text[:1000])


def test_fresh_rag_retrieval():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    # --------------------------------------------------------
    # Verify vector database
    # --------------------------------------------------------

    assert retriever.index.ntotal == 260761

    assert retriever.index.d == 1024

    assert len(retriever.metadata) == 260761

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    queries = [
        "What is the standard rate of sales tax in Pakistan?",

        "What is input tax under the Sales Tax Act 1990?",

        "What are the conditions for claiming input tax?",

        "What is section 8B of the Sales Tax Act 1990?",

        "What is the sales tax treatment of zero rated supplies?",
    ]

    print()
    print("=" * 100)
    print("ZABTA FRESH RAG RETRIEVAL TEST")
    print("=" * 100)

    print(
        f"Vector count: {retriever.index.ntotal:,}"
    )

    print(
        f"Embedding dimension: {retriever.index.d}"
    )

    print(
        f"Metadata records: {len(retriever.metadata):,}"
    )

    for query in queries:

        results = retriever.search(
            query,
            top_k=5,
        )

        assert results

        assert len(results) == 5

        print_results(
            query,
            results,
        )

    print()
    print("=" * 100)
    print("✓ FRESH RAG RETRIEVAL TEST PASSED")
    print("=" * 100)