from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)


def test_current_rate_service_retrieve():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=5,
    )

    results = service.retrieve(
        "What is the standard sales tax rate in Pakistan?"
    )

    assert isinstance(
        results,
        list,
    )

    assert len(results) > 0

    assert "text" in results[0]

    assert "metadata" in results[0]