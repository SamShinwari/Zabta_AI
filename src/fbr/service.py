from __future__ import annotations

from pathlib import Path

from src.fbr.agent import FBRAgent
from src.fbr.generator import FBRGenerator
from src.fbr.ollama_backend import OllamaBackend
from src.fbr.qa import FBRQA
from src.fbr.reranker import FBRReranker
from src.fbr.retriever import FBRRetriever


DEFAULT_VECTOR_DIR = (
    "data/vector_database/fbr"
)


class ZabtaFBRService:
    """
    Main FBR question-answering service for Zabta.

    Pipeline:

        User Question
              ↓
        FBRRetriever
              ↓
        FBRReranker
              ↓
        FBRQA
              ↓
        FBRGenerator
              ↓
        OllamaBackend
              ↓
        Llama 3.1 8B
              ↓
        Final FBR Answer + Sources
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        vector_dir: str | Path = DEFAULT_VECTOR_DIR,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        temperature: float = 0.0,
    ):
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k must be greater than zero"
            )

        if final_top_k > retrieval_top_k:
            raise ValueError(
                "final_top_k cannot be greater than "
                "retrieval_top_k"
            )

        # ----------------------------------------------------
        # 1. FBR Retriever
        # ----------------------------------------------------

        self.retriever = FBRRetriever(
            vector_dir=vector_dir,
            embedding_model="BAAI/bge-m3",
        )

        # ----------------------------------------------------
        # 2. FBR Reranker
        # ----------------------------------------------------

        self.reranker = FBRReranker()

        # ----------------------------------------------------
        # 3. FBR QA
        # ----------------------------------------------------

        self.qa = FBRQA(
            retriever=self.retriever,
            reranker=self.reranker,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
        )

        # ----------------------------------------------------
        # 4. Ollama backend
        # ----------------------------------------------------

        self.backend = OllamaBackend(
            model=model,
            temperature=temperature,
        )

        # ----------------------------------------------------
        # 5. FBR Generator
        # ----------------------------------------------------

        self.generator = FBRGenerator(
            model=model,
            backend=self.backend,
        )

        # ----------------------------------------------------
        # 6. Complete FBR Agent
        # ----------------------------------------------------

        self.agent = FBRAgent(
            qa=self.qa,
            generator=self.generator,
        )

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
    ):
        """
        Ask Zabta an FBR question.

        Returns:
            FBRAgentResponse
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string"
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty"
            )

        return self.agent.ask(
            question
        )

    # ========================================================
    # PREPARE
    # ========================================================

    def prepare(
        self,
        question: str,
    ) -> dict:
        """
        Run retrieval, reranking, citation and
        prompt preparation without calling the LLM.
        """

        return self.agent.prepare(
            question
        )

    # ========================================================
    # INFO
    # ========================================================

    def info(self) -> dict:
        """
        Return service configuration.
        """

        return {
            "service": "ZabtaFBRService",
            "model": self.generator.model,
            "vector_directory": str(
                self.retriever.vector_dir
            ),
            "vector_count": int(
                self.retriever.index.ntotal
            ),
            "embedding_dimension": int(
                self.retriever.index.d
            ),
            "retrieval_top_k": (
                self.qa.retrieval_top_k
            ),
            "final_top_k": (
                self.qa.final_top_k
            ),
            "temperature": (
                self.backend.temperature
            ),
        }