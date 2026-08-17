from src.fbr.ollama_backend import OllamaBackend


def test_ollama_backend():
    backend = OllamaBackend(
        model="llama3.1:8b",
        temperature=0,
    )

    response = backend(
        "What is 2 + 2? Answer briefly."
    )

    assert isinstance(
        response,
        str,
    )

    assert response.strip()

    print("\nOllama response:")
    print(response)

    print("\nBackend information:")
    print(backend.info())
