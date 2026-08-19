from __future__ import annotations

from pathlib import Path
from typing import Any

from src.fbr.retriever import FBRRetriever
from src.fbr.reranker import FBRReranker
from src.fbr.qa import FBRQA
from src.fbr.generator import FBRGenerator


class FBRPipeline:
    """
    End-to-end FBR question-answering pipeline.

    Pipeline:

        Query
          ↓
        Retriever
          ↓
        Reranker
          ↓
        QA
          ↓
        Generator
          ↓
        Answer + Citations
    """

    def __init__(
        self,
        vector_dir: str | Path,
        embedding_model: str = "BAAI/bge-m3",
        top_k: int = 10,
        rerank_k: int = 5,
    ):
        self.top_k = top_k
        self.rerank_k = rerank_k

        # --------------------------------------------------
        # Retriever
        # --------------------------------------------------

        self.retriever = FBRRetriever(
            vector_dir=vector_dir,
            embedding_model=embedding_model,
        )

        # --------------------------------------------------
        # Reranker
        # --------------------------------------------------

        self.reranker = FBRReranker()

        # --------------------------------------------------
        # QA
        # --------------------------------------------------

        self.qa = FBRQA(retriever=self.retriever, reranker=self.reranker, retrieval_top_k=self.top_k, final_top_k=self.rerank_k)
        # --------------------------------------------------
        # Generator
        # --------------------------------------------------

        self.generator = FBRGenerator()

    # ======================================================
    # Ask
    # ======================================================

    def ask(
        self,
        question: str,
        current_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Answer an FBR-related question.

        Returns:

        {
            "question": "...",
            "answer": "...",
            "citations": [...],
            "retrieved": [...],
            "reranked": [...]
        }
        """

        if not isinstance(question, str):
            raise TypeError(
                "question must be a string"
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty"
            )

        # --------------------------------------------------
        # 1. Retrieve
        # --------------------------------------------------

        retrieved = self.retriever.search(
            question,
            top_k=self.top_k,
        )

        # --------------------------------------------------
        # 2. Rerank
        # --------------------------------------------------

        reranked = self.reranker.rerank(
            retrieved,
            current_year=current_year,
        )

        reranked = reranked[: self.rerank_k]

        # --------------------------------------------------
        # 3. QA processing
        # --------------------------------------------------

        qa_result = self.qa.answer(
            question,
            reranked,
        )

        # --------------------------------------------------
        # 4. Generate final answer
        # --------------------------------------------------
        if self.generator.backend is None:
            generated = qa_result["answer"]
        else:
            generated = self.generator.generate(question,qa_result,)    

        # --------------------------------------------------
        # 5. Normalize output
        # --------------------------------------------------

        if isinstance(generated, dict):

            answer = generated.get(
                "answer",
                generated.get(
                    "text",
                    "",
                ),
            )

            citations = generated.get(
                "citations",
                qa_result.get(
                    "citations",
                    [],
                )
                if isinstance(qa_result, dict)
                else [],
            )

        else:

            answer = str(generated)

            citations = (
                qa_result.get(
                    "citations",
                    [],
                )
                if isinstance(qa_result, dict)
                else []
            )

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved": retrieved,
            "reranked": reranked,
            "qa": qa_result,
        }