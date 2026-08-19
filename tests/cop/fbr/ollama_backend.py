from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama


class OllamaBackend:
    """
    Ollama LLM backend for the Zabta FBR QA system.

    This class adapts LangChain's ChatOllama interface
    to the callable backend expected by FBRGenerator.

    Flow:

        FBRGenerator
             ↓
        OllamaBackend
             ↓
        ChatOllama
             ↓
        Ollama server
             ↓
        llama3.1:8b
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        base_url: str = "http://127.0.0.1:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        self.llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    # ========================================================
    # CALL
    # ========================================================

    def __call__(
        self,
        prompt: str,
    ) -> str:
        """
        Send a prompt to Ollama and return the generated text.
        """

        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string"
            )

        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )

        response = self.llm.invoke(
            prompt
        )

        content = getattr(
            response,
            "content",
            None,
        )

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "Ollama response content must be a string"
            )

        content = content.strip()

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response"
            )

        return content

    # ========================================================
    # INVOKE
    # ========================================================

    def invoke(
        self,
        prompt: str,
    ) -> str:
        """
        Explicit invocation method.

        This is equivalent to calling the backend directly.
        """

        return self(
            prompt
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Return backend configuration information.
        """

        return {
            "backend": "ollama",
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }
