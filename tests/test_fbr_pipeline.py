from pathlib import Path

import pytest

from src.fbr.pipeline import FBRPipeline


VECTOR_DIR = Path("data/vector_database/fbr")


@pytest.fixture(scope="module")
def pipeline():
    return FBRPipeline(
        vector_dir=VECTOR_DIR,
        top_k=10,
        rerank_k=5,
    )


def test_pipeline_initializes(pipeline):
    assert pipeline.retriever is not None
    assert pipeline.reranker is not None
    assert pipeline.qa is not None
    assert pipeline.generator is not None


def test_pipeline_vector_database(pipeline):
    assert pipeline.retriever.vector_count > 0
    assert pipeline.retriever.dimension == 1024


def test_pipeline_top_k(pipeline):
    assert pipeline.top_k == 10
    assert pipeline.rerank_k == 5


def test_pipeline_empty_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("")


def test_pipeline_whitespace_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_pipeline_invalid_question(pipeline):
    with pytest.raises(TypeError):
        pipeline.ask(None)


def test_pipeline_question_is_string(pipeline):
    with pytest.raises(TypeError):
        pipeline.ask(123)


def test_pipeline_basic_question(pipeline):
    result = pipeline.ask(
        "What is the standard sales tax rate?"
    )

    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "citations" in result
    assert "retrieved" in result
    assert "reranked" in result
    assert "qa" in result


def test_pipeline_retrieval_results(pipeline):
    result = pipeline.ask(
        "What is the standard sales tax rate?"
    )

    assert len(result["retrieved"]) > 0


def test_pipeline_reranked_results(pipeline):
    result = pipeline.ask(
        "What is the standard sales tax rate?"
    )

    assert len(result["reranked"]) > 0
    assert len(result["reranked"]) <= 5


def test_pipeline_answer(pipeline):
    result = pipeline.ask(
        "What is the standard sales tax rate?"
    )

    assert isinstance(result["answer"], str)
    assert len(result["answer"].strip()) > 0


def test_pipeline_citations(pipeline):
    result = pipeline.ask(
        "What is the standard sales tax rate?"
    )

    assert isinstance(result["citations"], list)


def test_pipeline_question_preserved(pipeline):
    question = (
        "What is the standard sales tax rate?"
    )

    result = pipeline.ask(question)

    assert result["question"] == question