import pytest

from src.fbr.generator import FBRGenerator


def test_generator_creation():

    generator = FBRGenerator(
        model="llama3.1:8b"
    )

    assert generator.model == "llama3.1:8b"


def test_generate_with_backend():

    def fake_backend(prompt):

        return "The standard sales tax rate is 18%."

    generator = FBRGenerator(
        backend=fake_backend
    )

    answer = generator.generate(
        "What is the standard sales tax rate?"
    )

    assert answer == (
        "The standard sales tax rate is 18%."
    )


def test_empty_prompt():

    generator = FBRGenerator(
        backend=lambda prompt: "answer"
    )

    with pytest.raises(
        ValueError
    ):
        generator.generate("")


def test_non_string_prompt():

    generator = FBRGenerator(
        backend=lambda prompt: "answer"
    )

    with pytest.raises(
        TypeError
    ):
        generator.generate(123)


def test_missing_backend():

    generator = FBRGenerator()

    with pytest.raises(
        RuntimeError
    ):
        generator.generate(
            "What is sales tax?"
        )


def test_empty_backend_response():

    generator = FBRGenerator(
        backend=lambda prompt: ""
    )

    with pytest.raises(
        RuntimeError
    ):
        generator.generate(
            "What is sales tax?"
        )


def test_non_string_backend_response():

    generator = FBRGenerator(
        backend=lambda prompt: 123
    )

    with pytest.raises(
        TypeError
    ):
        generator.generate(
            "What is sales tax?"
        )


def test_generate_from_prepared():

    def fake_backend(prompt):

        assert "FBR CONTEXT" in prompt

        return "Grounded FBR answer."

    generator = FBRGenerator(
        backend=fake_backend
    )

    prepared = {
        "prompt": (
            "FBR CONTEXT\n"
            "Sales Tax Act, 1990\n"
        )
    }

    answer = generator.generate_from_prepared(
        prepared
    )

    assert answer == (
        "Grounded FBR answer."
    )
