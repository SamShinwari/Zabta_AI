from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass
class TaxRateCandidate:
    """
    A tax-rate candidate extracted from an FBR document.
    """

    rate: float
    source: str
    page: Any
    text: str
    authority_score: float
    semantic_score: float
    retrieval_score: float
    year: int | None = None
    category: str = "unknown"


class FBRRateResolver:
    """
    Resolve sales-tax rate candidates from retrieved
    FBR evidence.

    This component:

        1. extracts rates
        2. classifies rates
        3. ranks candidates
        4. selects the strongest applicable candidate

    It does NOT calculate invoice tax.
    """

    # ========================================================
    # ENGLISH NUMBER WORDS
    # ========================================================

    NUMBER_WORDS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "twenty-one": 21,
        "twenty-two": 22,
        "twenty-three": 23,
        "twenty-four": 24,
        "twenty-five": 25,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
    }

    # ========================================================
    # ENGLISH-NUMBER RATE EXTRACTION
    # ========================================================

    @classmethod
    def extract_word_rates(
        cls,
        text: str,
    ) -> list[float]:
        """
        Extract rates written using English number words.
        """

        if not isinstance(text, str):
            return []

        normalized = text.lower()

        normalized = normalized.replace(
            "[",
            " ",
        )

        normalized = normalized.replace(
            "]",
            " ",
        )

        rates: list[float] = []

        for word, value in cls.NUMBER_WORDS.items():

            pattern = (
                rf"\b{re.escape(word)}\b"
                rf"\s+(?:per\s+cent|percent)"
            )

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                rates.append(
                    float(value)
                )

        return rates

    # ========================================================
    # NUMERIC RATE PATTERNS
    # ========================================================

    RATE_PATTERNS = (
        r"rate\s+of\s+(\d+(?:\.\d+)?)\s*%",
        r"sales\s+tax\s+(?:at|of|rate\s+of)\s+"
        r"(\d+(?:\.\d+)?)\s*%",
        r"\bgst\s+"
        r"(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%",
    )

    # ========================================================
    # RATE EXTRACTION
    # ========================================================

    @classmethod
    def extract_rates(
        cls,
        text: str,
    ) -> list[float]:
        """
        Extract numeric and English-number percentage
        values from text.
        """

        if not isinstance(text, str):
            return []

        if not text.strip():
            return []

        rates: list[float] = []

        text_lower = text.lower()

        # ----------------------------------------------------
        # Numeric rates
        # ----------------------------------------------------

        for pattern in cls.RATE_PATTERNS:

            matches = re.findall(
                pattern,
                text_lower,
                flags=re.IGNORECASE,
            )

            for match in matches:

                if isinstance(
                    match,
                    tuple,
                ):
                    match = match[0]

                try:
                    rate = float(match)

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if 0 < rate <= 100:
                    rates.append(rate)

        # ----------------------------------------------------
        # Word rates
        # ----------------------------------------------------

        rates.extend(
            cls.extract_word_rates(
                text
            )
        )

        return sorted(
            set(rates)
        )

    # ========================================================
    # YEAR EXTRACTION
    # ========================================================

    @staticmethod
    def extract_year(
        source: str,
    ) -> int | None:
        """
        Extract the latest four-digit year from
        a document name.
        """

        if not isinstance(
            source,
            str,
        ):
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
    # RATE CLASSIFICATION
    # ========================================================

    @staticmethod
    def classify_rate(
        rate: float,
        text: str,
    ) -> str:
        """
        Classify a rate based on the surrounding FBR text.

        Categories:

            standard
            special
            reduced
            unknown
        """

        if not isinstance(
            text,
            str,
        ):
            return "unknown"

        text_lower = text.lower()

        # ----------------------------------------------------
        # Reduced rate
        # ----------------------------------------------------

        reduced_keywords = (
            "reduced rate",
            "reduced sales tax",
            "reduction in sales tax",
            "reduced tax rate",
            "at the reduced rate",
        )

        if any(
            keyword in text_lower
            for keyword in reduced_keywords
        ):
            return "reduced"

        # ----------------------------------------------------
        # Special / enhanced rate
        # ----------------------------------------------------

        special_keywords = (
            "enhanced rate",
            "special rate",
            "higher rate",
            "luxury goods",
            "specified goods",
            "five percent",
            "25% sales tax",
            "25 percent sales tax",
        )

        if any(
            keyword in text_lower
            for keyword in special_keywords
        ):
            return "special"

        # ----------------------------------------------------
        # Standard rate
        # ----------------------------------------------------

        standard_keywords = (
            "standard rate",
            "standard sales tax",
            "scope of tax",
            "taxable supplies",
            "sales tax at the rate",
            "sales tax shall be charged",
        )

        if any(
            keyword in text_lower
            for keyword in standard_keywords
        ):
            return "standard"

        return "unknown"

    # ========================================================
    # CANDIDATE EXTRACTION
    # ========================================================

    def extract_candidates(
        self,
        results: list[dict[str, Any]],
    ) -> list[TaxRateCandidate]:
        """
        Extract tax-rate candidates from retrieved
        FBR search results.
        """

        candidates: list[
            TaxRateCandidate
        ] = []

        for result in results:

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

            text = result.get(
                "text",
                "",
            )

            combined_text = (
                f"{source}\n{text}"
            )

            rates = self.extract_rates(
                text
            )

            if not rates:
                continue

            semantic_score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            authority_score = float(
                result.get(
                    "authority_score",
                    0.0,
                )
            )

            retrieval_score = float(
                result.get(
                    "retrieval_score",
                    semantic_score,
                )
            )

            year = self.extract_year(
                source
            )

            for rate in rates:

                category = self.classify_rate(
                    rate,
                    combined_text,
                )

                candidates.append(
                    TaxRateCandidate(
                        rate=rate,
                        source=source,
                        page=page,
                        text=text,
                        authority_score=authority_score,
                        semantic_score=semantic_score,
                        retrieval_score=retrieval_score,
                        year=year,
                        category=category,
                    )
                )

        return candidates

    # ========================================================
    # CANDIDATE RANKING
    # ========================================================

    def rank_candidates(
        self,
        candidates: list[TaxRateCandidate],
    ) -> list[TaxRateCandidate]:
        """
        Rank rate candidates.

        Ranking considers:

            1. document authority
            2. document recency
            3. retrieval relevance
        """

        current_year = date.today().year

        def candidate_score(
            candidate: TaxRateCandidate,
        ) -> float:

            # ------------------------------------------------
            # Recency
            # ------------------------------------------------

            year_score = 0.50

            if candidate.year is not None:

                age = max(
                    0,
                    current_year
                    - candidate.year,
                )

                year_score = max(
                    0.50,
                    1.0 - (
                        age * 0.05
                    ),
                )

            # ------------------------------------------------
            # Category adjustment
            # ------------------------------------------------

            category_score = {
                "standard": 1.00,
                "reduced": 0.90,
                "special": 0.80,
                "unknown": 0.70,
            }.get(
                candidate.category,
                0.70,
            )

            return (
                0.40
                * candidate.authority_score
                + 0.30
                * year_score
                + 0.20
                * candidate.retrieval_score
                + 0.10
                * category_score
            )

        return sorted(
            candidates,
            key=candidate_score,
            reverse=True,
        )

    # ========================================================
    # BASIC RESOLVE
    # ========================================================

    def resolve(
        self,
        results: list[dict[str, Any]],
    ) -> TaxRateCandidate | None:
        """
        Resolve the strongest tax-rate candidate.

        This is the low-level resolver.
        """

        candidates = (
            self.extract_candidates(
                results
            )
        )

        if not candidates:
            return None

        ranked = self.rank_candidates(
            candidates
        )

        return ranked[0]

    # ========================================================
    # RESOLVE FROM RESULTS
    # ========================================================

    def resolve_from_results(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Resolve the strongest rate while preserving
        the original FBR evidence.

        Returns:

            {
                "result": {
                    "rate": 18.0,
                    "category": "standard",
                    "confidence": 0.82,
                },
                "source": original_retrieved_result,
            }

        This method is used by FBRCurrentRateService.
        """

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            return None

        candidates = self.extract_candidates(
            results
        )

        if not candidates:
            return None

        ranked = self.rank_candidates(
            candidates
        )

        best = ranked[0]

        # ----------------------------------------------------
        # Find original retrieved result
        # ----------------------------------------------------

        source_result = None

        for result in results:

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

            text = result.get(
                "text",
                "",
            )

            if (
                source == best.source
                and page == best.page
                and text == best.text
            ):
                source_result = result
                break

        if source_result is None:
            source_result = {}

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            0.40 * best.authority_score
            + 0.35 * best.retrieval_score
        )

        if best.year is not None:

            current_year = date.today().year

            age = max(
                0,
                current_year - best.year,
            )

            recency = max(
                0.50,
                1.0 - (
                    age * 0.05
                ),
            )

            confidence += (
                0.25 * recency
            )

        confidence = min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

        return {
            "result": {
                "rate": best.rate,
                "category": best.category,
                "confidence": confidence,
                "year": best.year,
            },
            "source": source_result,
            "candidate": best,
        }