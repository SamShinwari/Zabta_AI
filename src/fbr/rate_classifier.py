from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RateClassification:
    """
    Classification of a tax-rate candidate.
    """

    rate: float
    category: str
    confidence: float
    reason: str


class FBRRateClassifier:
    """
    Classify sales-tax rate candidates using the
    surrounding FBR document text.

    Categories:

        standard
        special
        reduced
        unknown
    """

    # ========================================================
    # KEYWORDS
    # ========================================================

        # ========================================================
    # KEYWORDS
    # ========================================================

    STANDARD_KEYWORDS = (
        "standard rate",
        "standard rates",
        "general rate",
        "normal rate",
        "rate of sales tax",
        "sales tax at the rate",
        "tax at the rate",
        "scope of tax",
    )

    SPECIAL_KEYWORDS = (
        "special rate",
        "special rates",
        "enhanced rate",
        "additional rate",
        "further tax",
        "extra tax",
        "luxury goods",
    )

    REDUCED_KEYWORDS = (
        "reduced rate",
        "reduced rates",
        "lower rate",
        "concessionary rate",
        "concession",
        "reduction",
        "reduced sales tax",
    )
    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Normalize FBR extracted PDF text.
        """

        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Remove square brackets commonly produced
        # by amended FBR legislation.
        text = text.replace(
            "[",
            " ",
        )

        text = text.replace(
            "]",
            " ",
        )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(
        self,
        rate: float,
        text: str,
    ) -> RateClassification:
        """
        Classify a rate using its surrounding text.

        Explicit rate terminology has priority over
        generic product descriptions.
        """

        if not isinstance(
            rate,
            (int, float),
        ):
            raise TypeError(
                "rate must be numeric"
            )

        normalized = self.normalize_text(
            text
        )

        # ----------------------------------------------------
        # Reduced rate
        # ----------------------------------------------------

        for keyword in self.REDUCED_KEYWORDS:

            if keyword in normalized:

                return RateClassification(
                    rate=float(rate),
                    category="reduced",
                    confidence=0.90,
                    reason=(
                        f"Matched reduced-rate "
                        f"keyword: '{keyword}'"
                    ),
                )

        # ----------------------------------------------------
        # Special rate
        # ----------------------------------------------------

        for keyword in self.SPECIAL_KEYWORDS:

            if keyword in normalized:

                return RateClassification(
                    rate=float(rate),
                    category="special",
                    confidence=0.90,
                    reason=(
                        f"Matched special-rate "
                        f"keyword: '{keyword}'"
                    ),
                )

        # ----------------------------------------------------
        # Standard rate
        # ----------------------------------------------------

        for keyword in self.STANDARD_KEYWORDS:

            if keyword in normalized:

                return RateClassification(
                    rate=float(rate),
                    category="standard",
                    confidence=0.85,
                    reason=(
                        f"Matched standard-rate "
                        f"keyword: '{keyword}'"
                    ),
                )

        # ----------------------------------------------------
        # Unknown
        # ----------------------------------------------------

        return RateClassification(
            rate=float(rate),
            category="unknown",
            confidence=0.30,
            reason=(
                "No rate classification keyword matched"
            ),
        )
    # ========================================================
    # CLASSIFY RESULT
    # ========================================================

    def classify_result(
        self,
        result: dict[str, Any],
    ) -> list[RateClassification]:
        """
        Extract percentages from one retrieved result
        and classify each candidate.

        This method deliberately accepts a generic result
        dictionary so it can work directly with FBRRetriever.
        """

        text = result.get(
            "text",
            "",
        )

        rates = result.get(
            "rates",
            [],
        )

        if not rates:
            return []

        classifications = []

        for rate in rates:

            classifications.append(
                self.classify(
                    rate=rate,
                    text=text,
                )
            )

        return classifications