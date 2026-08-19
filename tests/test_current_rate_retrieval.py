from src.fbr.retriever import FBRRetriever


VECTOR_DIR = "data/vector_database/fbr"


def test_current_sales_tax_rate_retrieval():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    results = retriever.search(
        "What is the current standard sales tax rate in Pakistan?",
        top_k=10,
    )

    assert len(results) > 0

    print()
    print("=" * 80)
    print("CURRENT SALES TAX RATE RETRIEVAL")
    print("=" * 80)

    for result in results:

        source = result.get(
            "metadata",
            {},
        ).get(
            "source",
            "",
        )

        page = result.get(
            "metadata",
            {},
        ).get(
            "page",
            "",
        )

        print()
        print(
            f"Rank:              {result['rank']}"
        )

        print(
            f"FAISS score:       {result['score']:.4f}"
        )

        print(
            f"Authority score:   "
            f"{result.get('authority_score', 0):.4f}"
        )

        print(
            f"Retrieval score:   "
            f"{result.get('retrieval_score', 0):.4f}"
        )

        print(
            f"Source:            {source}"
        )

        print(
            f"Page:              {page}"
        )

        print(
            f"Text:              "
            f"{result.get('text', '')[:300]}"
        )

    print()