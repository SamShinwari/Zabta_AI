from __future__ import annotations

from typing import Any

from src.fbr.query_analyzer import FBRQueryAnalyzer
from src.fbr.rate_resolver import FBRRateResolver
from src.fbr.retriever import FBRRetriever
from src.fbr.current_rate import CurrentRateResult


class FBRCurrentRateService:
    """
    Resolve the currently applicable sales tax rate
    from retrieved FBR documents.

    Pipeline:

        Query
          ↓
        FBRRetriever
          ↓
        FBRQueryAnalyzer
          ↓
        FBRRateResolver
          ↓
        CurrentRateResult
    """

    def __init__(
        self,
        vector_dir: str = "data/vector_database/fbr",
        retrieval_top_k: int = 10,
    ):
        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        self.retriever = FBRRetriever(
            vector_dir=vector_dir,
            embedding_model="BAAI/bge-m3",
        )

        self.query_analyzer = FBRQueryAnalyzer()

        self.rate_resolver = FBRRateResolver()

        self.retrieval_top_k = retrieval_top_k

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        question: str,
    ) -> CurrentRateResult:
        """
        Resolve a sales tax rate from FBR evidence.
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

        # ----------------------------------------------------
        # 1. Retrieve FBR evidence
        # ----------------------------------------------------

        results = self.retriever.search(
            question,
            top_k=self.retrieval_top_k,
        )

        if not results:
            raise LookupError(
                "No FBR evidence was retrieved "
                "for the sales tax rate query."
            )

        # ----------------------------------------------------
        # 2. Resolve best rate
        # ----------------------------------------------------

        resolved = self.rate_resolver.resolve_from_results(
            results
        )

        if resolved is None:
            raise LookupError(
                "No usable sales tax rate could be "
                "resolved from the retrieved FBR evidence."
            )

        # ----------------------------------------------------
        # 3. Build result
        # ----------------------------------------------------

        result = resolved["result"]
        source = resolved["source"]

        metadata = source.get(
            "metadata",
            {},
        )

        return CurrentRateResult(
            rate=float(
                result["rate"]
            ),
            category=result.get(
                "category",
                "unknown",
            ),
            source_document=metadata.get(
                "source",
                "Unknown FBR document",
            ),
            page=metadata.get(
                "page"
            ),
            chunk=metadata.get(
                "chunk"
            ),
            confidence=float(
                result.get(
                    "confidence",
                    source.get(
                        "score",
                        0.0,
                    ),
                )
            ),
            text=source.get(
                "text",
                "",
            ),
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    def info(self) -> dict[str, Any]:
        """
        Return configuration information.
        """

        return {
            "service": "FBRCurrentRateService",
            "embedding_model": "BAAI/bge-m3",
            "vector_directory": str(
                self.retriever.vector_dir
            ),
            "vector_count": self.retriever.vector_count,
            "embedding_dimension": self.retriever.dimension,
            "retrieval_top_k": self.retrieval_top_k,
        }