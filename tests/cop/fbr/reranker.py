from __future__ import annotations

import re
from datetime import datetime
from typing import Any


class FBRReranker:
    """
    Re-rank FBR retrieval results using:

    1. Semantic similarity
    2. Document recency
    3. Document authority

    The original FAISS score is preserved.
    """

    # --------------------------------------------------------
    # Document authority
    # --------------------------------------------------------

    AUTHORITY_SCORES = {
        "sales tax act": 1.00,
        "sales tax rules": 0.95,
        "finance act": 0.90,
        "sro": 0.85,
        "notification": 0.80,
        "circular": 0.75,
        "knowledge base": 0.60,
    }

    def __init__(
        self,
        semantic_weight: float = 0.70,
        recency_weight: float = 0.20,
        authority_weight: float = 0.10,
    ):

        total = (
            semantic_weight
            + recency_weight
            + authority_weight
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Reranker weights must sum to 1.0"
            )

        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.authority_weight = authority_weight

    # ========================================================
    # Authority
    # ========================================================

    def authority_score(
        self,
        source: str,
    ) -> float:

        source_lower = source.lower()

        for document_type, score in (
            self.AUTHORITY_SCORES.items()
        ):

            if document_type in source_lower:
                return score

        return 0.50

    # ========================================================
    # Year extraction
    # ========================================================

    def extract_year(
        self,
        source: str,
    ) -> int | None:

        years = re.findall(
            r"\b(19\d{2}|20\d{2})\b",
            source,
        )

        if not years:
            return None

        return max(
            int(year)
            for year in years
        )

    # ========================================================
    # Recency
    # ========================================================

    def recency_score(
        self,
        source: str,
        current_year: int | None = None,
    ) -> float:

        if current_year is None:
            current_year = datetime.now().year

        year = self.extract_year(source)

        if year is None:
            return 0.50

        age = max(
            0,
            current_year - year,
        )

        # Newer documents receive higher scores.
        #
        # Current year      -> 1.00
        # 1 year old        -> 0.95
        # 2 years old       -> 0.90
        # ...
        # 10+ years old     -> 0.50

        return max(
            0.50,
            1.0 - (age * 0.05),
        )

    # ========================================================
    # Final score
    # ========================================================

    def score_result(
        self,
        result: dict[str, Any],
        current_year: int | None = None,
    ) -> dict[str, Any]:

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "",
        )

        semantic_score = float(
            result.get(
                "score",
                0.0,
            )
        )

        authority = self.authority_score(
            source
        )

        recency = self.recency_score(
            source,
            current_year=current_year,
        )

        final_score = (
            self.semantic_weight
            * semantic_score
            + self.recency_weight
            * recency
            + self.authority_weight
            * authority
        )

        reranked = dict(result)

        reranked[
            "semantic_score"
        ] = semantic_score

        reranked[
            "authority_score"
        ] = authority

        reranked[
            "recency_score"
        ] = recency

        reranked[
            "rerank_score"
        ] = final_score

        return reranked

    # ========================================================
    # Re-rank
    # ========================================================

    def rerank(
        self,
        results: list[dict[str, Any]],
        current_year: int | None = None,
    ) -> list[dict[str, Any]]:

        scored_results = [
            self.score_result(
                result,
                current_year=current_year,
            )
            for result in results
        ]

        scored_results.sort(
            key=lambda result: result[
                "rerank_score"
            ],
            reverse=True,
        )

        for rank, result in enumerate(
            scored_results,
            start=1,
        ):

            result["rank"] = rank

        return scored_results
