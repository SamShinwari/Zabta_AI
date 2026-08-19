from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ApplicableTaxRate:
    """
    Represents the rate selected for an invoice item.

    A retrieved FBR percentage is not automatically
    considered the applicable invoice rate.
    """

    base_rate: float | None
    additional_rate: float | None
    category: str
    confidence: float
    source: str
    page: Any
    explanation: str


class FBRRateApplicabilityResolver:
    """
    Determine which retrieved tax-rate candidate represents
    the base sales-tax rate for an invoice.

    Important:

        standard/base rate
            !=
        further tax

    Further tax is kept separately and must not replace
    the base sales-tax rate.
    """

    # ========================================================
    # MAIN RESOLUTION
    # ========================================================

    def resolve(
        self,
        candidates: list[dict[str, Any]],
    ) -> ApplicableTaxRate:

        if not isinstance(candidates, list):
            raise TypeError(
                "candidates must be a list"
            )

        if not candidates:
            return ApplicableTaxRate(
                base_rate=None,
                additional_rate=None,
                category="unknown",
                confidence=0.0,
                source="",
                page=None,
                explanation=(
                    "No tax-rate candidates were provided."
                ),
            )

        # ----------------------------------------------------
        # Normalize candidates
        # ----------------------------------------------------

        normalized = []

        for candidate in candidates:

            if not isinstance(candidate, dict):
                continue

            rate = candidate.get("rate")

            if rate is None:
                continue

            try:
                rate = float(rate)
            except (
                TypeError,
                ValueError,
            ):
                continue

            category = str(
                candidate.get(
                    "category",
                    "unknown",
                )
            ).lower()

            normalized.append(
                {
                    **candidate,
                    "rate": rate,
                    "category": category,
                }
            )

        if not normalized:
            return ApplicableTaxRate(
                base_rate=None,
                additional_rate=None,
                category="unknown",
                confidence=0.0,
                source="",
                page=None,
                explanation=(
                    "No valid tax-rate candidates were found."
                ),
            )

        # ====================================================
        # 1. FIND STANDARD / BASE RATE
        # ====================================================

        standard_candidates = [
            candidate
            for candidate in normalized
            if candidate["category"]
            in {
                "standard",
                "base",
                "general",
            }
        ]

        # ====================================================
        # 2. FIND FURTHER TAX
        # ====================================================

        further_candidates = [
            candidate
            for candidate in normalized
            if candidate["category"]
            in {
                "further",
                "further_tax",
            }
        ]

        # ====================================================
        # 3. SELECT BASE RATE
        # ====================================================

        if standard_candidates:

            standard_candidates.sort(
                key=lambda candidate: (
                    float(
                        candidate.get(
                            "confidence",
                            candidate.get(
                                "retrieval_score",
                                candidate.get(
                                    "score",
                                    0.0,
                                ),
                            ),
                        )
                    ),
                    float(
                        candidate.get(
                            "year",
                            0,
                        ) or 0
                    ),
                ),
                reverse=True,
            )

            selected = standard_candidates[0]

            additional_rate = None

            if further_candidates:

                further_candidates.sort(
                    key=lambda candidate: (
                        float(
                            candidate.get(
                                "confidence",
                                candidate.get(
                                    "retrieval_score",
                                    candidate.get(
                                        "score",
                                        0.0,
                                    ),
                                ),
                            )
                        ),
                        float(
                            candidate.get(
                                "year",
                                0,
                            ) or 0
                        ),
                    ),
                    reverse=True,
                )

                additional_rate = (
                    further_candidates[0]["rate"]
                )

            confidence = float(
                selected.get(
                    "confidence",
                    selected.get(
                        "retrieval_score",
                        selected.get(
                            "score",
                            0.0,
                        ),
                    ),
                )
            )

            return ApplicableTaxRate(
                base_rate=selected["rate"],
                additional_rate=additional_rate,
                category="standard",
                confidence=confidence,
                source=str(
                    selected.get(
                        "source",
                        "",
                    )
                ),
                page=selected.get(
                    "page"
                ),
                explanation=(
                    f"{selected['rate']}% was selected as "
                    "the base sales-tax rate. Any further "
                    "tax candidate is kept separately and "
                    "is not treated as the base rate."
                ),
            )

        # ====================================================
        # 4. SPECIAL / REDUCED / ENHANCED
        # ====================================================

        priority_categories = (
            "reduced",
            "enhanced",
            "special",
            "zero-rated",
            "zero_rated",
            "exempt",
        )

        special_candidates = [
            candidate
            for candidate in normalized
            if candidate["category"]
            in priority_categories
        ]

        if special_candidates:

            special_candidates.sort(
                key=lambda candidate: float(
                    candidate.get(
                        "confidence",
                        candidate.get(
                            "retrieval_score",
                            candidate.get(
                                "score",
                                0.0,
                            ),
                        ),
                    )
                ),
                reverse=True,
            )

            selected = special_candidates[0]

            confidence = float(
                selected.get(
                    "confidence",
                    selected.get(
                        "retrieval_score",
                        selected.get(
                            "score",
                            0.0,
                        ),
                    ),
                )
            )

            return ApplicableTaxRate(
                base_rate=selected["rate"],
                additional_rate=None,
                category=selected["category"],
                confidence=confidence,
                source=str(
                    selected.get(
                        "source",
                        "",
                    )
                ),
                page=selected.get(
                    "page"
                ),
                explanation=(
                    "A special tax-rate category was "
                    "identified from the retrieved FBR evidence."
                ),
            )

        # ====================================================
        # 5. NO APPLICABLE BASE RATE
        # ====================================================

        return ApplicableTaxRate(
            base_rate=None,
            additional_rate=(
                further_candidates[0]["rate"]
                if further_candidates
                else None
            ),
            category="unknown",
            confidence=0.0,
            source="",
            page=None,
            explanation=(
                "The retrieved evidence contains no clearly "
                "identified base sales-tax rate."
            ),
        )