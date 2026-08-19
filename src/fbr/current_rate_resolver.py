from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.fbr.rate_classifier import (
    FBRRateClassifier,
)


@dataclass
class CurrentRateResult:
    """
    Resolved current sales-tax rate.

    The result keeps the evidence that caused the
    rate to be selected.
    """

    rate: float
    category: str
    source: str
    page: Any
    effective_date: date | None
    confidence: float
    reason: str
    text: str


class CurrentRateResolver:
    """
    Resolve the currently applicable sales-tax rate
    from retrieved FBR evidence.

    IMPORTANT:

        This class does not calculate invoice tax.

        It determines which rate should be passed
        to the tax calculator.
    """

    # ========================================================
    # SOURCE AUTHORITY
    # ========================================================

    AUTHORITY = {
        "sales tax act": 1.00,
        "sales tax rules": 0.95,
        "finance act": 0.90,
        "sro": 0.85,
        "notification": 0.80,
        "circular": 0.75,
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        classifier: FBRRateClassifier | None = None,
    ):
        self.classifier = (
            classifier
            if classifier is not None
            else FBRRateClassifier()
        )

    # ========================================================
    # AUTHORITY
    # ========================================================

    def authority_score(
        self,
        source: str,
    ) -> float:
        """
        Determine legal-document authority.
        """

        source_lower = source.lower()

        for document_type, score in (
            self.AUTHORITY.items()
        ):
            if document_type in source_lower:
                return score

        return 0.40

    # ========================================================
    # YEAR EXTRACTION
    # ========================================================

    @staticmethod
    def extract_year(
        source: str,
    ) -> int | None:
        """
        Extract the latest year appearing in
        the source filename.
        """

        if not isinstance(source, str):
            return None

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
    # DOCUMENT DATE
    # ========================================================

    @classmethod
    def document_date(
        cls,
        source: str,
    ) -> date | None:
        """
        Extract common dates from FBR filenames.

        Supported examples:

            30-06-2025
            30.06.2025
            30th June, 2024
            11th March, 2019
        """

        if not isinstance(source, str):
            return None

        # ----------------------------------------------------
        # DD-MM-YYYY
        # ----------------------------------------------------

        match = re.search(
            r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",
            source,
        )

        if match:

            day, month, year = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

            try:
                return date(
                    year,
                    month,
                    day,
                )
            except ValueError:
                pass

        # ----------------------------------------------------
        # DD.MM.YYYY
        # ----------------------------------------------------

        match = re.search(
            r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b",
            source,
        )

        if match:

            day, month, year = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

            try:
                return date(
                    year,
                    month,
                    day,
                )
            except ValueError:
                pass

        # ----------------------------------------------------
        # Month names
        # ----------------------------------------------------

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        month_pattern = (
            r"\b(\d{1,2})(?:st|nd|rd|th)?"
            r"\s+([A-Za-z]+)"
            r"(?:,)?\s+(\d{4})\b"
        )

        match = re.search(
            month_pattern,
            source,
        )

        if match:

            day = int(
                match.group(1)
            )

            month_name = match.group(
                2
            ).lower()

            year = int(
                match.group(3)
            )

            month = months.get(
                month_name
            )

            if month is not None:

                try:
                    return date(
                        year,
                        month,
                        day,
                    )
                except ValueError:
                    pass

        return None

    # ========================================================
    # RECENCY
    # ========================================================

    @staticmethod
    def recency_score(
        document_date: date | None,
        as_of: date,
    ) -> float:
        """
        Score evidence based on how recent the
        document is relative to the requested date.
        """

        if document_date is None:
            return 0.50

        if document_date > as_of:
            return 0.00

        days_old = (
            as_of - document_date
        ).days

        # Approximate yearly decay.
        years_old = days_old / 365.25

        return max(
            0.50,
            1.00 - (
                years_old * 0.05
            ),
        )

    # ========================================================
    # CANDIDATES
    # ========================================================

    def build_candidates(
        self,
        results: list[dict[str, Any]],
        as_of: date | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert retrieved RAG results into structured
        rate candidates.
        """

        if as_of is None:
            as_of = date.today()

        candidates = []

        for result in results:

            text = result.get(
                "text",
                "",
            )

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "",
            )

            page = metadata.get(
                "page",
                None,
            )

            # ------------------------------------------------
            # Extract rates
            # ------------------------------------------------

            rates = result.get(
                "rates",
                [],
            )

            if not rates:

                # Import locally to keep this resolver
                # independently testable.
                from src.fbr.rate_resolver import (
                    FBRRateResolver,
                )

                rates = (
                    FBRRateResolver.extract_rates(
                        text
                    )
                )

            if not rates:
                continue

            # ------------------------------------------------
            # Document metadata
            # ------------------------------------------------

            doc_date = (
                self.document_date(
                    source
                )
            )

            authority = (
                self.authority_score(
                    source
                )
            )

            semantic = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            rerank_score = float(
                result.get(
                    "rerank_score",
                    semantic,
                )
            )

            recency = (
                self.recency_score(
                    doc_date,
                    as_of,
                )
            )

            # ------------------------------------------------
            # Future document protection
            # ------------------------------------------------

            if (
                doc_date is not None
                and doc_date > as_of
            ):
                continue

            # ------------------------------------------------
            # Rate classification
            # ------------------------------------------------

            for rate in rates:

                classification = (
                    self.classifier.classify(
                        rate=rate,
                        text=text,
                    )
                )

                candidates.append(
                    {
                        "rate": float(rate),
                        "category": (
                            classification.category
                        ),
                        "classification_confidence": (
                            classification.confidence
                        ),
                        "classification_reason": (
                            classification.reason
                        ),
                        "source": source,
                        "page": page,
                        "text": text,
                        "document_date": doc_date,
                        "authority_score": authority,
                        "semantic_score": semantic,
                        "rerank_score": rerank_score,
                        "recency_score": recency,
                    }
                )

        return candidates

    # ========================================================
    # RANK
    # ========================================================

    @staticmethod
    def _candidate_score(
        candidate: dict[str, Any],
    ) -> float:
        """
        Calculate deterministic candidate score.
        """

        authority = float(
            candidate.get(
                "authority_score",
                0.0,
            )
        )

        recency = float(
            candidate.get(
                "recency_score",
                0.0,
            )
        )

        retrieval = float(
            candidate.get(
                "rerank_score",
                candidate.get(
                    "semantic_score",
                    0.0,
                ),
            )
        )

        classification = float(
            candidate.get(
                "classification_confidence",
                0.0,
            )
        )

        # Legal authority receives the largest weight.
        return (
            0.35 * authority
            + 0.30 * recency
            + 0.20 * retrieval
            + 0.15 * classification
        )

    def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Rank structured rate candidates.
        """

        ranked = []

        for candidate in candidates:

            item = dict(
                candidate
            )

            item["resolution_score"] = (
                self._candidate_score(
                    candidate
                )
            )

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda item: item[
                "resolution_score"
            ],
            reverse=True,
        )

        return ranked

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        results: list[dict[str, Any]],
        as_of: date | None = None,
        category: str = "standard",
    ) -> CurrentRateResult | None:
        """
        Resolve the best currently applicable rate.

        By default, only STANDARD rates are considered.

        This is intentional.

        A 25% special rate must not automatically replace
        the standard rate of 18%.
        """

        if as_of is None:
            as_of = date.today()

        candidates = self.build_candidates(
            results,
            as_of=as_of,
        )

        if not candidates:
            return None

        # ----------------------------------------------------
        # Filter by requested category
        # ----------------------------------------------------

        if category:

            category_candidates = [
                candidate
                for candidate in candidates
                if candidate[
                    "category"
                ] == category
            ]

            # If the requested category exists,
            # use only those candidates.
            if category_candidates:
                candidates = (
                    category_candidates
                )

        # ----------------------------------------------------
        # Rank
        # ----------------------------------------------------

        ranked = self.rank_candidates(
            candidates
        )

        if not ranked:
            return None

        best = ranked[0]

        return CurrentRateResult(
            rate=best["rate"],
            category=best["category"],
            source=best["source"],
            page=best["page"],
            effective_date=best[
                "document_date"
            ],
            confidence=best[
                "resolution_score"
            ],
            reason=(
                "Selected highest-ranked "
                f"{best['category']} rate using "
                "authority, recency, retrieval "
                "relevance, and classification "
                "confidence."
            ),
            text=best["text"],
        )