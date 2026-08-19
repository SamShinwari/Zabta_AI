from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
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
    applicability: str = "unknown"
    context: str = ""

    product_match_score: float = 0.0
    hs_code_match: bool = False

    effective_from: str | None = None
    effective_to: str | None = None

    date_relevance_score: float = 0.0


class FBRRateResolver:
    """
    Resolve sales-tax rate candidates from retrieved
    FBR evidence.

    This component:

        1. extracts rates
        2. classifies rates
        3. checks product / HS-code applicability
        4. checks document date relevance
        5. ranks candidates
        6. selects the strongest candidate

    It does NOT calculate invoice tax.

    Important:

        Document cutoff dates are retrieval signals.
        They are NOT automatically legal effective dates.
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

        Examples:

            "eighteen per cent"
                -> [18.0]

            "[eighteen] per cent"
                -> [18.0]

            "twenty-five percent"
                -> [25.0]
        """

        if not isinstance(text, str):
            return []

        normalized = text.lower()

        # ----------------------------------------------------
        # Normalize OCR brackets
        # ----------------------------------------------------

        normalized = normalized.replace(
            "[",
            " ",
        )

        normalized = normalized.replace(
            "]",
            " ",
        )

        rates: list[float] = []

        # ----------------------------------------------------
        # Extract word-based percentages
        # ----------------------------------------------------

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
        # Example:
        # "rate of 18%"
        r"rate\s+of\s+(\d+(?:\.\d+)?)\s*%",

        # Example:
        # "sales tax at 18%"
        r"sales\s+tax\s+(?:at|of|rate\s+of)\s+"
        r"(\d+(?:\.\d+)?)\s*%",

        # Example:
        # "GST 18%"
        r"\bgst\s+"
        r"(\d+(?:\.\d+)?)\s*%",

        # Generic percentage:
        # "subject to 25%"
        # "at 18%"
        # "tax = 17%"
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

                # re.findall() may return tuples
                # if the regex contains multiple groups.
                if isinstance(match, tuple):
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

        # ----------------------------------------------------
        # English-number rates
        # ----------------------------------------------------

        word_rates = cls.extract_word_rates(
            text
        )

        rates.extend(
            word_rates
        )

        # ----------------------------------------------------
        # Remove duplicates
        # ----------------------------------------------------

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
    # DOCUMENT CUTOFF DATE
    # ========================================================

    @staticmethod
    def extract_cutoff_date(
        source: str,
    ) -> str | None:
        """
        Extract a document amendment/cutoff date
        from its filename.

        Examples:

            "Sales Tax Act amended upto 30-06-2025"
                -> "2025-06-30"

            "Sales Tax Act amended up to 30th June, 2024"
                -> "2024-06-30"

            "Sales Tax Act amended upto 11th March, 2019"
                -> "2019-03-11"

        Returns:
            ISO date string when a recognizable date
            is found, otherwise None.

        Important:
            This is a document cutoff/amendment date,
            NOT necessarily the legal effective date
            of every provision contained in the document.
        """

        if not isinstance(
            source,
            str,
        ):
            return None

        text = source.lower()

        # ----------------------------------------------------
        # Numeric date
        #
        # 30-06-2025
        # 30.06.2025
        # 30/06/2025
        # ----------------------------------------------------

        match = re.search(
            r"(?:amended\s+)?"
            r"(?:upto|up\s+to)"
            r"\s+"
            r"(\d{1,2})"
            r"[-./]"
            r"(\d{1,2})"
            r"[-./]"
            r"(19\d{2}|20\d{2})",
            text,
        )

        if match:

            day = int(
                match.group(1)
            )

            month = int(
                match.group(2)
            )

            year = int(
                match.group(3)
            )

            try:
                parsed_date = date(
                    year,
                    month,
                    day,
                )

            except ValueError:
                return None

            return parsed_date.isoformat()

        # ----------------------------------------------------
        # ISO-style date
        #
        # 2025-06-30
        # 2025.06.30
        # 2025/06/30
        # ----------------------------------------------------

        match = re.search(
            r"(19\d{2}|20\d{2})"
            r"[-./]"
            r"(\d{1,2})"
            r"[-./]"
            r"(\d{1,2})",
            text,
        )

        if match:

            year = int(
                match.group(1)
            )

            month = int(
                match.group(2)
            )

            day = int(
                match.group(3)
            )

            try:
                parsed_date = date(
                    year,
                    month,
                    day,
                )

            except ValueError:
                return None

            return parsed_date.isoformat()

        # ----------------------------------------------------
        # Month-name dates
        #
        # 30th June, 2024
        # 11th March, 2019
        # 30 June 2024
        # ----------------------------------------------------

        month_names = {
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
            "|".join(
                month_names.keys()
            )
        )

        match = re.search(
            rf"(\d{{1,2}})"
            rf"(?:st|nd|rd|th)?"
            rf"\s+"
            rf"({month_pattern})"
            rf"(?:,)?"
            rf"\s+"
            rf"(19\d{{2}}|20\d{{2}})",
            text,
        )

        if match:

            day = int(
                match.group(1)
            )

            month = month_names[
                match.group(2)
            ]

            year = int(
                match.group(3)
            )

            try:
                parsed_date = date(
                    year,
                    month,
                    day,
                )

            except ValueError:
                return None

            return parsed_date.isoformat()

        return None

    # ========================================================
    # DATE RELEVANCE
    # ========================================================
    @staticmethod
    def is_date_eligible(
        candidate: TaxRateCandidate,
        invoice_date: str | None = None,
    ) -> bool:
        """
        Determine whether a document candidate is eligible
        for the invoice date.

        A document whose amendment/cutoff date is after
        the invoice date must not be used to determine
        the invoice's applicable rate.

        Important:
            This is a temporal retrieval filter.
            It does NOT prove the legal effective date
            of an individual tax provision.
        """

        # ----------------------------------------------------
        # No invoice date
        # ----------------------------------------------------

        if not invoice_date:
            return True

        # ----------------------------------------------------
        # No document cutoff date
        # ----------------------------------------------------

        if not candidate.effective_from:
            return True

        try:
            invoice = datetime.strptime(
                invoice_date,
                "%Y-%m-%d",
            ).date()

            cutoff = datetime.strptime(
                candidate.effective_from,
                "%Y-%m-%d",
            ).date()

        except ValueError:
            return True

        # ----------------------------------------------------
        # Future document
        # ----------------------------------------------------

        if cutoff > invoice:
            return False

        return True
    @staticmethod
    def calculate_date_relevance(
        candidate: TaxRateCandidate,
        invoice_date: str | None,
    ) -> float:
        """
        Calculate how relevant a candidate document is
        to the invoice date.

        This is a retrieval signal, NOT legal proof that
        the rate remained effective.
        """

        # ----------------------------------------------------
        # No invoice date
        # ----------------------------------------------------

        if not invoice_date:
            return 0.5

        # ----------------------------------------------------
        # Unknown document cutoff
        # ----------------------------------------------------

        if not candidate.effective_from:
            return 0.5

        try:

            invoice = datetime.strptime(
                invoice_date,
                "%Y-%m-%d",
            ).date()

            cutoff = datetime.strptime(
                candidate.effective_from,
                "%Y-%m-%d",
            ).date()

        except ValueError:
            return 0.5

        # ----------------------------------------------------
        # Document cutoff is after invoice date.
        #
        # This document can contain rules applicable
        # to the invoice date.
        # ----------------------------------------------------

        if cutoff <= invoice:
            return 1.0
        return 0.0

        # ----------------------------------------------------
        # Older documents receive a decreasing score.
        # ----------------------------------------------------

        age_days = (
            invoice - cutoff
        ).days

        age_years = (
            age_days / 365.25
        )

        return max(
            0.0,
            1.0 - (
                age_years * 0.20
            ),
        )

    # ========================================================
    # RATE CONTEXT CLASSIFICATION
    # ========================================================

    @staticmethod
    def classify_rate_context(
        rate: float,
        text: str,
    ) -> tuple[str, str]:
        """
        Classify a tax rate based on its surrounding
        FBR document context.

        Returns:

            (
                category,
                applicability,
            )
        """

        if not isinstance(
            text,
            str,
        ):
            return (
                "unknown",
                "unknown",
            )

        normalized = text.lower()

        # ----------------------------------------------------
        # Normalize OCR brackets
        # ----------------------------------------------------

        normalized = normalized.replace(
            "[",
            " ",
        )

        normalized = normalized.replace(
            "]",
            " ",
        )

        # ----------------------------------------------------
        # Exempt
        # ----------------------------------------------------

        if (
            "exempt from sales tax" in normalized
            or "exempted from sales tax" in normalized
        ):
            return (
                "exempt",
                "conditional",
            )

        # ----------------------------------------------------
        # Zero-rated
        # ----------------------------------------------------

        zero_rated_patterns = (
            "zero-rated",
            "zero rated",
            "zero rate",
            "zero rates",
        )

        if any(
            pattern in normalized
            for pattern in zero_rated_patterns
        ):
            return (
                "zero-rated",
                "conditional",
            )

        # ----------------------------------------------------
        # Further tax
        # ----------------------------------------------------

        further_patterns = (
            "further tax",
            "further sales tax",
            "further tax at the rate",
        )

        if any(
            pattern in normalized
            for pattern in further_patterns
        ):
            return (
                "further",
                "conditional",
            )

        # ----------------------------------------------------
        # Reduced rate
        # ----------------------------------------------------

        reduced_patterns = (
            "reduced rate",
            "reduced rates",
            "reduced sales tax",
            "reduction in sales tax",
            "reduced tax rate",
            "at the reduced rate",
            "tax at the rate of five",
            "tax at the rate of 5",
        )

        if any(
            pattern in normalized
            for pattern in reduced_patterns
        ):
            return (
                "reduced",
                "conditional",
            )

        # ----------------------------------------------------
        # Enhanced rate
        # ----------------------------------------------------

        enhanced_patterns = (
            "enhanced rate",
            "enhanced rates",
            "increased rate",
            "increased rates",
            "higher rate",
        )

        if any(
            pattern in normalized
            for pattern in enhanced_patterns
        ):
            return (
                "enhanced",
                "conditional",
            )

        # ----------------------------------------------------
        # Special rate
        # ----------------------------------------------------

        special_patterns = (
            "special rate",
            "special rates",
            "specified goods",
            "specified supplies",
            "special provision",
            "luxury goods",
        )

        if any(
            pattern in normalized
            for pattern in special_patterns
        ):
            return (
                "special",
                "conditional",
            )

        # ----------------------------------------------------
        # Standard rate
        # ----------------------------------------------------

        standard_patterns = (
            "tax known as sales tax at the rate",
            "sales tax at the rate",
            "sales tax shall be charged",
            "shall be charged, levied and paid a tax",
            "standard rate",
            "standard sales tax",
            "scope of tax",
            "taxable supplies",
        )

        if any(
            pattern in normalized
            for pattern in standard_patterns
        ):
            return (
                "standard",
                "general",
            )

        # ----------------------------------------------------
        # Unknown
        # ----------------------------------------------------

        return (
            "unknown",
            "unknown",
        )

    # ========================================================
    # HS CODE MATCHING
    # ========================================================

    @staticmethod
    def hs_code_match(
        text: str,
        hs_code: str | None,
    ) -> bool:
        """
        Check whether the retrieved FBR text explicitly
        contains the requested HS code.
        """

        if not hs_code:
            return False

        if not isinstance(
            text,
            str,
        ):
            return False

        normalized_text = (
            text.lower()
            .replace("-", ".")
            .replace("/", ".")
        )

        normalized_code = (
            str(hs_code)
            .strip()
            .lower()
            .replace("-", ".")
            .replace("/", ".")
        )

        return normalized_code in normalized_text

    # ========================================================
    # PRODUCT MATCHING
    # ========================================================

    @staticmethod
    def product_match_score(
        text: str,
        item_description: str | None,
    ) -> float:
        """
        Estimate whether the retrieved FBR text discusses
        the requested invoice item.

        This is intentionally simple at this stage.
        """

        if not item_description:
            return 0.0

        if not isinstance(
            text,
            str,
        ):
            return 0.0

        text_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )

        item_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                item_description.lower(),
            )
        )

        if not item_words:
            return 0.0

        matches = text_words.intersection(
            item_words
        )

        return len(matches) / len(
            item_words
        )

    # ========================================================
    # CANDIDATE EXTRACTION
    # ========================================================

    def extract_candidates(
        self,
        results: list[dict[str, Any]],
        item_description: str | None = None,
        hs_code: str | None = None,
    ) -> list[TaxRateCandidate]:
        """
        Extract tax-rate candidates from retrieved
        FBR search results.

        Optional invoice information:

            item_description
            hs_code

        is used for applicability signals.
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

            # ------------------------------------------------
            # Combine source and text
            # ------------------------------------------------

            combined_text = (
                f"{source}\n{text}"
            )

            rates = self.extract_rates(
                text
            )

            if not rates:
                continue

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Document metadata
            # ------------------------------------------------

            year = self.extract_year(
                source
            )

            effective_from = (
                self.extract_cutoff_date(
                    source
                )
            )

            # ------------------------------------------------
            # Invoice applicability
            # ------------------------------------------------

            hs_code_match = (
                self.hs_code_match(
                    text=combined_text,
                    hs_code=hs_code,
                )
            )

            product_match_score = (
                self.product_match_score(
                    text=combined_text,
                    item_description=item_description,
                )
            )

            # ------------------------------------------------
            # Create one candidate for every rate
            # ------------------------------------------------

            for rate in rates:

                category, applicability = (
                    self.classify_rate_context(
                        rate=rate,
                        text=combined_text,
                    )
                )

                candidates.append(
                    TaxRateCandidate(
                        rate=rate,
                        source=source,
                        page=page,
                        text=text,
                        authority_score=(
                            authority_score
                        ),
                        semantic_score=(
                            semantic_score
                        ),
                        retrieval_score=(
                            retrieval_score
                        ),
                        year=year,
                        category=category,
                        applicability=applicability,
                        context=text,
                        product_match_score=(
                            product_match_score
                        ),
                        hs_code_match=(
                            hs_code_match
                        ),
                        effective_from=(
                            effective_from
                        ),
                    )
                )

        return candidates

    # ========================================================
    # CANDIDATE RANKING
    # ========================================================

    def rank_candidates(
        self,
        candidates: list[TaxRateCandidate],
        invoice_date: str | None = None,
    ) -> list[TaxRateCandidate]:
        """
        Rank rate candidates.

        Ranking considers:

            1. FBR document authority
            2. Document recency
            3. Retrieval relevance
            4. Invoice-date relevance

        Weights:

            Authority          = 35%
            Document recency   = 25%
            Retrieval           = 20%
            Date relevance      = 20%

        Date relevance is a retrieval signal only.
        It is NOT legal proof that a rate was effective
        on the invoice date.
        """
        # ----------------------------------------------------
        # Remove documents that did not exist yet on the
        # invoice date.
        # ----------------------------------------------------

        eligible_candidates = [
            candidate
            for candidate in candidates
            if self.is_date_eligible(
                candidate,
                invoice_date,
            )
        ]

        # ----------------------------------------------------
        # If every candidate is future-dated, keep the
        # original candidates as evidence rather than
        # returning an empty result.
        # ----------------------------------------------------

        if eligible_candidates:
            candidates = eligible_candidates
                # ----------------------------------------------------
        # Prefer the latest applicable FBR document.
        #
        # For tax-rate resolution, an older amendment
        # must not defeat a newer applicable amendment
        # merely because its semantic retrieval score
        # is higher.
        # ----------------------------------------------------

        dated_candidates = [
            candidate
            for candidate in candidates
            if candidate.effective_from
        ]

        if dated_candidates:
            latest_date = max(
                candidate.effective_from
                for candidate in dated_candidates
            )

            latest_candidates = [
                candidate
                for candidate in candidates
                if candidate.effective_from
                == latest_date
            ]

            if latest_candidates:
                candidates = latest_candidates
        current_year = date.today().year

        def candidate_score(
            candidate: TaxRateCandidate,
        ) -> float:

            # ------------------------------------------------
            # Document recency
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
            # Invoice-date relevance
            # ------------------------------------------------

            date_relevance = (
                self.calculate_date_relevance(
                    candidate,
                    invoice_date,
                )
            )

            candidate.date_relevance_score = (
                date_relevance
            )

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            return (
                0.35
                * candidate.authority_score
                + 0.25
                * year_score
                + 0.20
                * candidate.retrieval_score
                + 0.20
                * date_relevance
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
        invoice_date: str | None = None,
        item_description: str | None = None,
        hs_code: str | None = None,
    ) -> TaxRateCandidate | None:
        """
        Resolve the strongest tax-rate candidate.

        Optional invoice information:

            invoice_date
            item_description
            hs_code

        is used to improve candidate ranking.

        invoice_date is a retrieval relevance signal,
        not legal proof of applicability.
        """

        candidates = self.extract_candidates(
            results,
            item_description=item_description,
            hs_code=hs_code,
        )

        if not candidates:
            return None

        ranked = self.rank_candidates(
            candidates,
            invoice_date=invoice_date,
        )

        return ranked[0]

    # ========================================================
    # RESOLVE FROM RESULTS
    # ========================================================

    def resolve_from_results(
        self,
        results: list[dict[str, Any]],
        invoice_date: str | None = None,
        item_description: str | None = None,
        hs_code: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Resolve the strongest rate while preserving
        the original FBR evidence.

        Optional invoice information:

            invoice_date
            item_description
            hs_code

        is used during candidate extraction and ranking.
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

        # ----------------------------------------------------
        # Extract candidates
        # ----------------------------------------------------

        candidates = self.extract_candidates(
            results,
            item_description=item_description,
            hs_code=hs_code,
        )

        if not candidates:
            return None

        # ----------------------------------------------------
        # Rank candidates
        # ----------------------------------------------------

        ranked = self.rank_candidates(
            candidates,
            invoice_date=invoice_date,
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
            0.40
            * best.authority_score
            + 0.35
            * best.retrieval_score
        )

        # ----------------------------------------------------
        # Document recency
        # ----------------------------------------------------

        if best.year is not None:

            current_year = date.today().year

            age = max(
                0,
                current_year
                - best.year,
            )

            recency = max(
                0.50,
                1.0 - (
                    age * 0.05
                ),
            )

            confidence += (
                0.25
                * recency
            )

        # ----------------------------------------------------
        # Clamp confidence
        # ----------------------------------------------------

        confidence = min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return {
            "result": {
                "rate": best.rate,
                "category": best.category,
                "applicability": (
                    best.applicability
                ),
                "product_match_score": (
                    best.product_match_score
                ),
                "hs_code_match": (
                    best.hs_code_match
                ),
                "effective_from": (
                    best.effective_from
                ),
                "effective_to": (
                    best.effective_to
                ),
                "date_relevance_score": (
                    best.date_relevance_score
                ),
                "confidence": confidence,
                "year": best.year,
            },
            "source": source_result,
            "candidate": best,
        }

           # ========================================================
    # RESOLVE STANDARD RATE FROM RESULTS
    # ========================================================

    def resolve_standard_from_results(
        self,
        results: list[dict[str, Any]],
        invoice_date: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Resolve the strongest STANDARD/base sales-tax
        rate from FBR evidence.

        Only candidates classified as:

            category == "standard"

        are considered.

        Zero-rated, further, reduced, enhanced, exempt,
        special, and unknown candidates are excluded.

        Important:

            This method does NOT claim that the selected
            rate is legally effective merely because the
            document cutoff date is relevant.

            Date filtering/ranking is a retrieval signal.
        """

        # ----------------------------------------------------
        # Validate results
        # ----------------------------------------------------

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list"
            )

        if not results:
            return None

        # ----------------------------------------------------
        # Extract all candidates
        # ----------------------------------------------------

        candidates = self.extract_candidates(
            results
        )

        if not candidates:
            return None

        # ----------------------------------------------------
        # Keep ONLY standard/base candidates
        # ----------------------------------------------------

        standard_candidates = [
            candidate
            for candidate in candidates
            if candidate.category == "standard"
        ]

        if not standard_candidates:
            return None

        # ----------------------------------------------------
        # Date-aware ranking
        # ----------------------------------------------------

        ranked = self.rank_candidates(
            standard_candidates,
            invoice_date=invoice_date,
        )

        if not ranked:
            return None

        best = ranked[0]

        # ----------------------------------------------------
        # Find original FBR evidence
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
            0.40
            * best.authority_score
            + 0.35
            * best.retrieval_score
        )

        # ----------------------------------------------------
        # Recency
        # ----------------------------------------------------

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
                0.25
                * recency
            )

        # ----------------------------------------------------
        # Date relevance
        # ----------------------------------------------------

        confidence += (
            0.10
            * best.date_relevance_score
        )

        # ----------------------------------------------------
        # Clamp confidence
        # ----------------------------------------------------

        confidence = min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "result": {
                "rate": best.rate,

                "category": best.category,

                "applicability": (
                    best.applicability
                ),

                "effective_from": (
                    best.effective_from
                ),

                "effective_to": (
                    best.effective_to
                ),

                "date_relevance_score": (
                    best.date_relevance_score
                ),

                "product_match_score": (
                    best.product_match_score
                ),

                "hs_code_match": (
                    best.hs_code_match
                ),

                "confidence": confidence,

                "year": best.year,
            },

            "source": source_result,

            "candidate": best,
        } 