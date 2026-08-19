from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RateChangeCandidate:
    """
    Represents a possible FBR document that changes,
    reduces, enhances, exempts, or otherwise modifies
    a sales-tax rate.
    """

    rate: float | None
    category: str
    source: str
    page: Any
    text: str
    authority_score: float
    retrieval_score: float
    year: int | None = None
    change_detected: bool = False


class FBRRateChangeDetector:
    """
    Detect possible sales-tax rate changes from FBR evidence.

    This component does NOT make the final legal decision.

    It identifies documents containing language such as:

        enhanced rate
        reduced rate
        sales tax reduced
        sales tax increased
        rate of 25%
        rate of 5%
        exemption
        zero-rated
        rescinded
        amended
        substituted
        revised rate
        new rate
    """

    # ========================================================
    # CHANGE KEYWORDS
    # ========================================================

    CHANGE_PATTERNS = (

        # ----------------------------------------------------
        # Enhanced / increased rates
        # ----------------------------------------------------

        r"\benhanced\s+rate\b",
        r"\bincreased\s+rate\b",
        r"\bincrease\s+in\s+sales\s+tax\b",
        r"\bsales\s+tax\s+increased\b",
        r"\bsales\s+tax\s+enhanced\b",

        # ----------------------------------------------------
        # Reduced rates
        # ----------------------------------------------------

        r"\breduced\s+rate\b",
        r"\breduction\s+in\s+sales\s+tax\b",
        r"\bsales\s+tax\s+reduced\b",
        r"\bsales\s+tax\s+reduced\s+to\b",
        r"\breduced\s+to\s+\d+(?:\.\d+)?\s*%",
        r"\breduced\s+from\s+\d+(?:\.\d+)?\s*%\s+to\b",

        # ----------------------------------------------------
        # General rate-change language
        # ----------------------------------------------------

        r"\brevised\s+rate\b",
        r"\bnew\s+rate\b",
        r"\bsales\s+tax\s+rate\b",
        r"\brate\s+of\s+\d+(?:\.\d+)?\s*%",
        r"\brate\s+of\s+\[[^\]]+\]\s+per\s+cent\b",

        # ----------------------------------------------------
        # Legal modification language
        # ----------------------------------------------------

        r"\bsubstituted\b",
        r"\bamended\b",
        r"\brescinded\b",
        r"\bwithdrawn\b",

        # ----------------------------------------------------
        # Special treatment
        # ----------------------------------------------------

        r"\bzero[-\s]?rated\b",
        r"\bexempt\b",
    )

    # ========================================================
    # DOCUMENT TYPES
    # ========================================================

    CHANGE_DOCUMENT_KEYWORDS = (
        "sro",
        "notification",
        "finance act",
        "sales tax notification",
        "sales tax",
    )

    # ========================================================
    # RATE EXTRACTION
    # ========================================================

    RATE_PATTERNS = (
        # Example:
        # "rate of 18%"
        r"rate\s+of\s+(\d+(?:\.\d+)?)\s*%",

        # Example:
        # "sales tax at 18%"
        r"sales\s+tax\s+(?:at|of|rate\s+of)\s+"
        r"(\d+(?:\.\d+)?)\s*%",

        # Generic percentage:
        # "18%"
        r"(\d+(?:\.\d+)?)\s*%",
    )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:
        pass

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    @classmethod
    def contains_change_language(
        cls,
        text: str,
    ) -> bool:
        """
        Determine whether text contains language suggesting
        a sales-tax change or modification.
        """

        if not isinstance(
            text,
            str,
        ):
            return False

        normalized = text.lower()

        for pattern in cls.CHANGE_PATTERNS:

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    # ========================================================
    # RATE EXTRACTION
    # ========================================================

    @classmethod
    def extract_rates(
        cls,
        text: str,
    ) -> list[float]:
        """
        Extract numeric percentage rates from text.
        """

        if not isinstance(
            text,
            str,
        ):
            return []

        rates: list[float] = []

        normalized = text.lower()

        for pattern in cls.RATE_PATTERNS:

            matches = re.findall(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            for match in matches:

                # re.findall() may return tuples
                # when a pattern contains multiple groups.
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

                # ------------------------------------------------
                # Sanity check
                # ------------------------------------------------

                if 0 < rate <= 100:
                    rates.append(rate)

        return sorted(
            set(rates)
        )

    # ========================================================
    # CATEGORY
    # ========================================================

    @staticmethod
    def classify_change(
        text: str,
    ) -> str:
        """
        Classify the type of possible tax-rate change.
        """

        if not isinstance(
            text,
            str,
        ):
            return "possible_change"

        normalized = text.lower()

        # ----------------------------------------------------
        # Enhanced
        # ----------------------------------------------------

        if (
            "enhanced rate" in normalized
            or "increased rate" in normalized
            or "increase in sales tax" in normalized
            or "sales tax increased" in normalized
            or "sales tax enhanced" in normalized
        ):
            return "enhanced"

        # ----------------------------------------------------
        # Reduced
        # ----------------------------------------------------

        if (
            "reduced rate" in normalized
            or "reduction in sales tax" in normalized
            or "sales tax reduced" in normalized
            or "reduced to" in normalized
        ):
            return "reduced"

        # ----------------------------------------------------
        # Zero rated
        # ----------------------------------------------------

        if (
            "zero-rated" in normalized
            or "zero rated" in normalized
        ):
            return "zero_rated"

        # ----------------------------------------------------
        # Exempt
        # ----------------------------------------------------

        if "exempt" in normalized:
            return "exempt"

        # ----------------------------------------------------
        # Rescinded / withdrawn
        # ----------------------------------------------------

        if (
            "rescinded" in normalized
            or "withdrawn" in normalized
        ):
            return "rescinded"

        # ----------------------------------------------------
        # Amended / substituted
        # ----------------------------------------------------

        if (
            "substituted" in normalized
            or "amended" in normalized
            or "revised rate" in normalized
            or "new rate" in normalized
        ):
            return "amended"

        # ----------------------------------------------------
        # Unknown possible change
        # ----------------------------------------------------

        return "possible_change"

    # ========================================================
    # YEAR EXTRACTION
    # ========================================================

    @staticmethod
    def extract_year(
        source: str,
    ) -> int | None:
        """
        Extract the latest four-digit year from
        a document source name.
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
    # DOCUMENT TYPE
    # ========================================================

    @classmethod
    def is_change_document(
        cls,
        source: str,
    ) -> bool:
        """
        Determine whether a source appears to be a document
        capable of containing a tax-rate change.
        """

        if not isinstance(
            source,
            str,
        ):
            return False

        normalized = source.lower()

        return any(
            keyword in normalized
            for keyword in cls.CHANGE_DOCUMENT_KEYWORDS
        )

    # ========================================================
    # CANDIDATE EXTRACTION
    # ========================================================

    def extract_candidates(
        self,
        results: list[dict[str, Any]],
    ) -> list[RateChangeCandidate]:
        """
        Extract possible rate-change candidates from
        retrieved FBR results.
        """

        candidates: list[
            RateChangeCandidate
        ] = []

        for result in results:

            # ------------------------------------------------
            # Retrieved text
            # ------------------------------------------------

            text = result.get(
                "text",
                "",
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "",
            )

            # ------------------------------------------------
            # Change-language check
            # ------------------------------------------------

            if not self.contains_change_language(
                text
            ):
                continue

            # ------------------------------------------------
            # Extract rates
            # ------------------------------------------------

            rates = self.extract_rates(
                text
            )

            # ------------------------------------------------
            # Authority score
            # ------------------------------------------------

            authority_score = float(
                result.get(
                    "authority_score",
                    0.0,
                )
            )

            # ------------------------------------------------
            # Retrieval score
            # ------------------------------------------------

            retrieval_score = float(
                result.get(
                    "retrieval_score",
                    result.get(
                        "score",
                        0.0,
                    ),
                )
            )

            # ------------------------------------------------
            # Classify change
            # ------------------------------------------------

            category = self.classify_change(
                text
            )

            # ------------------------------------------------
            # Extract year
            # ------------------------------------------------

            year = self.extract_year(
                source
            )

            # ------------------------------------------------
            # Create candidates
            # ------------------------------------------------

            for rate in rates:

                candidates.append(
                    RateChangeCandidate(
                        rate=rate,
                        category=category,
                        source=source,
                        page=metadata.get(
                            "page"
                        ),
                        text=text,
                        authority_score=(
                            authority_score
                        ),
                        retrieval_score=(
                            retrieval_score
                        ),
                        year=year,
                        change_detected=True,
                    )
                )

        return candidates

    # ========================================================
    # RANK
    # ========================================================

    def rank_candidates(
        self,
        candidates: list[RateChangeCandidate],
    ) -> list[RateChangeCandidate]:
        """
        Rank possible rate-change documents.

        Authority and retrieval relevance are currently
        the main signals.
        """

        def score(
            candidate: RateChangeCandidate,
        ) -> float:

            return (
                0.60
                * candidate.authority_score
                + 0.40
                * candidate.retrieval_score
            )

        return sorted(
            candidates,
            key=score,
            reverse=True,
        )

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        results: list[dict[str, Any]],
    ) -> list[RateChangeCandidate]:
        """
        Detect and rank possible sales-tax changes.
        """

        candidates = self.extract_candidates(
            results
        )

        if not candidates:
            return []

        return self.rank_candidates(
            candidates
        )