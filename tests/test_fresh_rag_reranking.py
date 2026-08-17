from __future__ import annotations

from src.fbr.retriever import FBRRetriever
from src.fbr.reranker import FBRReranker


VECTOR_DIR = "data/vector_database/fbr"


def print_results(
    title: str,
    results: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print(f"RESULT {i}")
        print("-" * 100)

        print(
            f"Source: {metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: {metadata.get('page', 'Unknown')}"
        )

        print(
            f"Semantic score: "
            f"{result.get('semantic_score', result.get('score', 0)):.4f}"
        )

        if "rerank_score" in result:

            print(
                f"Recency score: "
                f"{result.get('recency_score', 0):.4f}"
            )

            print(
                f"Authority score: "
                f"{result.get('authority_score', 0):.4f}"
            )

            print(
                f"Rerank score: "
                f"{result.get('rerank_score', 0):.4f}"
            )

        text = result.get(
            "text",
            "",
        ).strip()

        print()
        print(text[:500])


def test_fresh_rag_reranking():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    reranker = FBRReranker()

    query = (
        "What is the standard rate of sales tax "
        "in Pakistan?"
    )

    # --------------------------------------------------------
    # 1. Semantic retrieval
    # --------------------------------------------------------

    retrieved = retriever.search(
        query,
        top_k=10,
    )

    assert len(retrieved) == 10

    print_results(
        "RAW FAISS RESULTS",
        retrieved[:5],
    )

    # --------------------------------------------------------
    # 2. Reranking
    # --------------------------------------------------------

    reranked = reranker.rerank(
        retrieved,
        current_year=2026,
    )

    assert len(reranked) == 10

    print_results(
        "RERANKED RESULTS",
        reranked[:5],
    )

    # --------------------------------------------------------
    # 3. Basic validation
    # --------------------------------------------------------

    assert "rerank_score" in reranked[0]

    assert "semantic_score" in reranked[0]

    assert "recency_score" in reranked[0]

    assert "authority_score" in reranked[0]

    # --------------------------------------------------------
    # 4. Show final winner
    # --------------------------------------------------------

    top_source = reranked[0][
        "metadata"
    ].get(
        "source",
        "",
    )

    print()
    print("=" * 100)
    print("FINAL RERANKED SOURCE")
    print("=" * 100)
    print(top_source)

    print()
    print("✓ Retrieval + reranking test passed")