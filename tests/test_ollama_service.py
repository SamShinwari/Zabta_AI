from src.fbr.service import ZabtaFBRService


def test_zabta_fbr_service():
    service = ZabtaFBRService(
        model="llama3.1:8b",
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=10,
        final_top_k=5,
        temperature=0,
    )

    print("\n" + "=" * 80)
    print("ZABTA SERVICE INFORMATION")
    print("=" * 80)

    print(service.info())

    question = (
        "What is the standard sales tax rate in Pakistan?"
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    response = service.ask(
        question
    )

    assert response.answer
    assert isinstance(
        response.answer,
        str,
    )

    assert response.sources
    assert response.retrieved_count > 0
    assert response.reranked_count > 0

    print("\n" + "=" * 80)
    print("FINAL ZABTA ANSWER")
    print("=" * 80)

    print(response.answer)

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in response.sources:
        print(
            "-",
            source.get(
                "citation",
                "Unknown source",
            ),
        )

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(
        "Retrieved:",
        response.retrieved_count,
    )

    print(
        "Reranked:",
        response.reranked_count,
    )