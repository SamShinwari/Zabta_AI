from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.fbr.citation import FBRCitationBuilder
from src.fbr.reranker import FBRReranker
from src.fbr.retriever import FBRRetriever


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class FBRAnswer:
    """
    Final answer structure returned by the FBR QA pipeline.
    """

    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_count: int
    reranked_count: int


# ============================================================
# FBR QA SYSTEM
# ============================================================


class FBRQA:
    """
    Retrieval-based QA pipeline for FBR documents.

    Pipeline:

        Question
            ↓
        Retriever
            ↓
        Reranker
            ↓
        Citation Builder
            ↓
        Context
            ↓
        Prompt
            ↓
        LLM / Evidence Answer
    """

    def __init__(
        self,
        retriever: FBRRetriever,
        reranker: FBRReranker | None = None,
        citation_builder: FBRCitationBuilder | None = None,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        current_year: int | None = None,
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

        self.retriever = retriever

        self.reranker = (
            reranker
            if reranker is not None
            else FBRReranker()
        )

        self.citation_builder = (
            citation_builder
            if citation_builder is not None
            else FBRCitationBuilder()
        )

        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.current_year = current_year

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_question(
        question: str,
    ) -> None:
        """
        Validate the user question.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string"
            )

        if not question.strip():
            raise ValueError(
                "question cannot be empty"
            )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidate FBR documents.
        """

        self._validate_question(
            question
        )

        return self.retriever.search(
            question,
            top_k=self.retrieval_top_k,
        )

    # ========================================================
    # RERANKING
    # ========================================================

    def rerank(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Rerank retrieved documents and keep the
        strongest final results.
        """

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            return []

        reranked = self.reranker.rerank(
            results,
            current_year=self.current_year,
        )

        return reranked[
            : self.final_top_k
        ]

    # ========================================================
    # CITATIONS
    # ========================================================

    def build_sources(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build clean FBR citations.

        Duplicate source/page combinations are removed
        by FBRCitationBuilder.build_many().
        """

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        return self.citation_builder.build_many(
            results
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    def build_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Build grounded context for the LLM.
        """

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            return ""

        context_parts: list[str] = []

        for number, result in enumerate(
            results,
            start=1,
        ):
            text = result.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "Unknown page",
            )

            chunk = metadata.get(
                "chunk",
                "Unknown chunk",
            )

            context_parts.append(
                f"[SOURCE {number}]\n"
                f"Document: {source}\n"
                f"Page: {page}\n"
                f"Chunk: {chunk}\n"
                f"Text:\n{text}"
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # PROMPT
    # ========================================================

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build a grounded prompt for the generation model.
        """

        self._validate_question(
            question
        )

        if not isinstance(
            context,
            str,
        ):
            raise TypeError(
                "context must be a string"
            )

        return (
            "You are an FBR sales tax compliance assistant.\n"
            "\n"
            "Answer the user's question using ONLY the "
            "provided FBR context.\n"
            "\n"
            "Rules:\n"
            "1. Do not invent legal requirements.\n"
            "2. Do not use information outside the context.\n"
            "3. If the context is insufficient, clearly say so.\n"
            "4. Prefer the most recent applicable document.\n"
            "5. Distinguish between Acts, Rules, SROs, "
            "notifications, and explanatory documents when "
            "possible.\n"
            "6. Give a concise and clear answer.\n"
            "7. Cite the relevant source numbers.\n"
            "\n"
            f"USER QUESTION:\n{question}\n"
            "\n"
            f"FBR CONTEXT:\n{context}\n"
            "\n"
            "ANSWER:"
        )

    # ========================================================
    # PREPARE
    # ========================================================

    def prepare(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Run the complete retrieval preparation pipeline.

        No LLM is called here.
        """

        self._validate_question(
            question
        )

        question = question.strip()

        # ----------------------------------------------------
        # 1. Retrieve
        # ----------------------------------------------------

        retrieved = self.retrieve(
            question
        )

        # ----------------------------------------------------
        # 2. Rerank
        # ----------------------------------------------------

        reranked = self.rerank(
            retrieved
        )

        # ----------------------------------------------------
        # 3. Citations
        # ----------------------------------------------------

        sources = self.build_sources(
            reranked
        )

        # ----------------------------------------------------
        # 4. Context
        # ----------------------------------------------------

        context = self.build_context(
            reranked
        )

        # ----------------------------------------------------
        # 5. Prompt
        # ----------------------------------------------------

        prompt = self.build_prompt(
            question,
            context,
        )

        return {
            "question": question,
            "retrieved": retrieved,
            "reranked": reranked,
            "sources": sources,
            "context": context,
            "prompt": prompt,
            "retrieved_count": len(
                retrieved
            ),
            "reranked_count": len(
                reranked
            ),
        }

    # ========================================================
    # EVIDENCE ANSWER
    # ========================================================

    def answer_from_context(
        self,
        question: str,
    ) -> FBRAnswer:
        """
        Return an evidence-backed response without
        calling an LLM.
        """

        prepared = self.prepare(
            question
        )

        reranked = prepared[
            "reranked"
        ]

        sources = prepared[
            "sources"
        ]

        if not reranked:
            answer_text = (
                "I could not find sufficient FBR evidence "
                "to answer this question."
            )

        else:
            evidence_parts: list[str] = []

            for number, result in enumerate(
                reranked,
                start=1,
            ):
                text = result.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    continue

                evidence_parts.append(
                    f"[Source {number}] {text}"
                )

            if evidence_parts:
                answer_text = "\n\n".join(
                    evidence_parts
                )
            else:
                answer_text = (
                    "I could not find sufficient textual "
                    "evidence in the retrieved FBR documents."
                )

        return FBRAnswer(
            question=prepared["question"],
            answer=answer_text,
            sources=sources,
            retrieved_count=prepared[
                "retrieved_count"
            ],
            reranked_count=prepared[
                "reranked_count"
            ],
        )

    # ========================================================
    # ANSWER
    # ========================================================

    def answer(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build an evidence-based answer from already
        retrieved and reranked FBR results.

        No LLM is called here.

        This method is used by FBRPipeline.
        """

        self._validate_question(
            question
        )

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        question = question.strip()

        # ----------------------------------------------------
        # Build citations
        # ----------------------------------------------------

        sources = self.build_sources(
            results
        )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = self.build_context(
            results
        )

        # ----------------------------------------------------
        # Build grounded prompt
        # ----------------------------------------------------

        prompt = self.build_prompt(
            question,
            context,
        )

        # ----------------------------------------------------
        # Evidence answer
        # ----------------------------------------------------

        if not results:
            answer_text = (
                "I could not find sufficient FBR evidence "
                "to answer this question."
            )

        else:
            evidence_parts: list[str] = []

            for number, result in enumerate(
                results,
                start=1,
            ):
                text = result.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    continue

                evidence_parts.append(
                    f"[Source {number}] {text}"
                )

            if evidence_parts:
                answer_text = "\n\n".join(
                    evidence_parts
                )
            else:
                answer_text = (
                    "I could not find sufficient textual "
                    "evidence in the retrieved FBR documents."
                )

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "citations": sources,
            "context": context,
            "prompt": prompt,
            "retrieved_count": len(
                results
            ),
            "reranked_count": len(
                results
            ),
        }

# ============================================================
# FACTORY
# ============================================================


def create_fbr_qa(
    retriever: FBRRetriever,
    reranker: FBRReranker | None = None,
    citation_builder: FBRCitationBuilder | None = None,
    retrieval_top_k: int = 10,
    final_top_k: int = 5,
    current_year: int | None = None,
) -> FBRQA:
    """
    Create and configure an FBRQA instance.

    This factory provides a simple public entry point for
    tests, the FBR pipeline, and application code.
    """

    return FBRQA(
        retriever=retriever,
        reranker=reranker,
        citation_builder=citation_builder,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k,
        current_year=current_year,
    )