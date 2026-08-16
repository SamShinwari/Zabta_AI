from pathlib import Path

import pytest

from src.fbr.qa import (
    FBRQA,
    create_fbr_qa,
)


VECTOR_DIR = Path(
    "data/vector_database/fbr"
)


def test_qa_import():

    assert FBRQA is not None


def test_create_fbr_qa():

    qa = create_fbr_qa(
        str(VECTOR_DIR),
        retrieval_top_k=5,
        final_top_k=3,
    )

    assert isinstance(
        qa,
        FBRQA,
    )


def test_invalid_retrieval_top_k():

    with pytest.raises(
        ValueError
    ):

        create_fbr_qa(
            str(VECTOR_DIR),
            retrieval_top_k=0,
        )


def test_invalid_final_top_k():

    with pytest.raises(
        ValueError
    ):

        create_fbr_qa(
            str(VECTOR_DIR),
            retrieval_top_k=5,
            final_top_k=10,
        )


def test_build_prompt():

    qa = create_fbr_qa(
        str(VECTOR_DIR),
        retrieval_top_k=5,
        final_top_k=3,
    )

    prompt = qa.build_prompt(
        "What is the standard sales tax rate?",
        "Sales tax is charged at the applicable rate.",
    )

    assert "standard sales tax rate" in prompt
    assert "FBR CONTEXT" in prompt


def test_empty_question():

    qa = create_fbr_qa(
        str(VECTOR_DIR),
        retrieval_top_k=5,
        final_top_k=3,
    )

    with pytest.raises(
        ValueError
    ):

        qa.retrieve("")