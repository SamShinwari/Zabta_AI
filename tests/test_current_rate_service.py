from src.fbr.current_rate_service import (
    FBRCurrentRateService,
)


def test_current_rate_service_initialization():

    service = FBRCurrentRateService(
        retrieval_top_k=10,
    )

    info = service.info()

    assert info["service"] == (
        "FBRCurrentRateService"
    )

    assert info["embedding_model"] == (
        "BAAI/bge-m3"
    )

    assert info["vector_count"] > 0

    assert info["embedding_dimension"] == 1024