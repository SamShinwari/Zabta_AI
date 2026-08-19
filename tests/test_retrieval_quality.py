from src.fbr.retriever import FBRRetriever


VECTOR_DIR = "data/vector_database/fbr"


def print_results(
    title: str,
    results: list[dict],
) -> None:

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

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
        print("-" * 90)

        print(
            "Score:",
            result["score"],
        )

        print(
            "Legal Match:",
            result.get(
                "legal_reference_match",
                False,
            ),
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "",
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "",
            ),
        )

        print(
            "Chunk:",
            metadata.get(
                "chunk",
                "",
            ),
        )

        print(
            "Text:",
            result["text"][:400].replace(
                "\n",
                " ",
            ),
        )


def test_section_retrieval_quality():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    query = (
        "What is section 8B of the Sales Tax Act 1990?"
    )

    results = retriever.search(
        query,
        top_k=10,
    )

    print_results(
        "SECTION 8B RETRIEVAL",
        results,
    )

    assert len(results) == 10

    legal_matches = [
        result
        for result in results
        if result.get(
            "legal_reference_match",
            False,
        )
    ]

    assert legal_matches

    print()
    print(
        "Legal-reference matches:",
        len(legal_matches),
    )


def test_sales_tax_rate_retrieval():

    retriever = FBRRetriever(
        vector_dir=VECTOR_DIR,
        embedding_model="BAAI/bge-m3",
    )

    query = (
        "What is the standard sales tax rate "
        "in Pakistan?"
    )

    results = retriever.search(
        query,
        top_k=10,
    )

    print_results(
        "STANDARD SALES TAX RATE RETRIEVAL",
        results,
    )

    assert len(results) == 10

    assert (
        "Sales Tax Act"
        in results[0]["metadata"]["source"]
    )