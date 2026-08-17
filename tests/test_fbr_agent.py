from __future__ import annotations

import pytest

from src.fbr.agent import (
    FBRAgent,
    FBRAgentResponse,
)
from src.fbr.generator import FBRGenerator
from src.fbr.qa import FBRQA


# ============================================================
# MOCK RETRIEVER
# ============================================================


class MockRetriever:

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        return [
            {
                "rank": 1,
                "score": 0.90,
                "text": (
                    "The standard sales tax rate is "
                    "eighteen per cent."
                ),
                "metadata": {
                    "source": (
                        "Sales Tax Act, 1990 "
                        "amended upto 30-06-2026.pdf"
                    ),
                    "page": 28,
                    "chunk": 1,
                },
            },
            {
                "rank": 2,
                "score": 0.80,
                "text": (
                    "Sales tax shall be charged "
                    "on taxable supplies."
                ),
                "metadata": {
                    "source": (
                        "Sales Tax Rules, 2006 "
                        "updated upto 31.10.2023.pdf"
                    ),
                    "page": 30,
                    "chunk": 2,
                },
            },
        ]


# ============================================================
# HELPERS
# ============================================================


def create_agent(
    backend=None,
):
    retriever = MockRetriever()

    qa = FBRQA(
        retriever=retriever,
        retrieval_top_k=2,
        final_top_k=2,
        current_year=2026,
    )

    generator = FBRGenerator(
        model="test-model",
        backend=backend,
    )

    return FBRAgent(
        qa=qa,
        generator=generator,
    )


# ============================================================
# TEST 1
# ============================================================


def test_agent_initialization():

    agent = create_agent(
        backend=lambda prompt: "Test answer"
    )

    assert isinstance(
        agent,
        FBRAgent,
    )


# ============================================================
# TEST 2
# ============================================================


def test_prepare():

    agent = create_agent(
        backend=lambda prompt: "Test answer"
    )

    prepared = agent.prepare(
        "What is the standard sales tax rate?"
    )

    assert (
        prepared["question"]
        == "What is the standard sales tax rate?"
    )

    assert prepared[
        "retrieved_count"
    ] == 2

    assert prepared[
        "reranked_count"
    ] == 2

    assert prepared[
        "context"
    ]

    assert prepared[
        "prompt"
    ]


# ============================================================
# TEST 3
# ============================================================


def test_generate():

    agent = create_agent(
        backend=lambda prompt: (
            "The standard sales tax rate is 18%."
        )
    )

    prepared = agent.prepare(
        "What is the standard sales tax rate?"
    )

    answer = agent.generate(
        prepared
    )

    assert (
        answer
        == "The standard sales tax rate is 18%."
    )


# ============================================================
# TEST 4
# ============================================================


def test_ask():

    agent = create_agent(
        backend=lambda prompt: (
            "The standard sales tax rate is 18%."
        )
    )

    response = agent.ask(
        "What is the standard sales tax rate?"
    )

    assert isinstance(
        response,
        FBRAgentResponse,
    )

    assert (
        response.answer
        == "The standard sales tax rate is 18%."
    )

    assert (
        response.retrieved_count
        == 2
    )

    assert (
        response.reranked_count
        == 2
    )

    assert len(
        response.sources
    ) > 0


# ============================================================
# TEST 5
# ============================================================


def test_ask_without_llm():

    agent = create_agent(
        backend=lambda prompt: "Should not be called"
    )

    response = agent.ask_without_llm(
        "What is the standard sales tax rate?"
    )

    assert response.question == (
        "What is the standard sales tax rate?"
    )

    assert response.retrieved_count == 2

    assert response.reranked_count == 2


# ============================================================
# TEST 6
# ============================================================


def test_format_response():

    agent = create_agent(
        backend=lambda prompt: "Test answer"
    )

    response = FBRAgentResponse(
        question="Test question",
        answer="The answer is 18%.",
        sources=[
            {
                "citation": (
                    "[1] Sales Tax Act, 1990, p. 28"
                )
            }
        ],
        retrieved_count=10,
        reranked_count=5,
    )

    formatted = agent.format_response(
        response
    )

    assert (
        "The answer is 18%."
        in formatted
    )

    assert (
        "[1] Sales Tax Act, 1990, p. 28"
        in formatted
    )


# ============================================================
# TEST 7
# ============================================================


def test_invalid_qa():

    generator = FBRGenerator(
        backend=lambda prompt: "answer"
    )

    with pytest.raises(
        TypeError
    ):
        FBRAgent(
            qa="invalid",
            generator=generator,
        )


# ============================================================
# TEST 8
# ============================================================


def test_invalid_generator():

    qa = FBRQA(
        retriever=MockRetriever()
    )

    with pytest.raises(
        TypeError
    ):
        FBRAgent(
            qa=qa,
            generator="invalid",
        )
def test_format_response_uses_application_citations():
    response = FBRAgentResponse(question="What is the standard sales tax rate?",answer=("The standard sales tax rate is 18%.\n\n""[ SOURCE 2 ]"),
        sources=[
            {
                "citation": (
                    "Sales Tax Act 1990 amended "
                    "upto 30-06-2025.pdf, p. 27"
                )
            },
        ],
        retrieved_count=10,
        reranked_count=5,
    )

    formatted = FBRAgent.format_response(response)

    assert "The standard sales tax rate is 18%." in formatted
    assert "[ SOURCE 2 ]" not in formatted
    assert "[1] Sales Tax Act 1990 amended upto 30-06-2025.pdf, p. 27" in formatted