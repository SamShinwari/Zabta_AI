from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.fbr.invoice_rate_query import InvoiceRateQuery
from src.fbr.rate_resolver import TaxRateCandidate
from src.fbr.rate_change_detector import (
    FBRRateChangeDetector,
)
from src.fbr.rate_applicability import (
    FBRRateApplicabilityResolver,
)


@dataclass
class InvoiceRateResolution:
    """
    Final sales-tax rate resolution for one invoice item.
    """

    item_description: str
    hs_code: str | None
    invoice_date: str | None

    rate: float | None
    category: str
    confidence: float

    source: str | None
    page: Any
    text: str

    query: str

    candidate: TaxRateCandidate | None = None

    # --------------------------------------------------------
    # Rate-change evidence
    # --------------------------------------------------------

    change_candidates: list[Any] | None = None

    # --------------------------------------------------------
    # Applicability information
    # --------------------------------------------------------

    additional_rate: float | None = None

    applicability_explanation: str = ""


class FBRInvoiceRateResolver:
    """
    Resolve the applicable sales-tax rate for an invoice item.

    Pipeline:

        Invoice item
             ↓
        InvoiceRateQuery
             ↓
        FBRCurrentRateService
             ↓
        FBRRetriever
             ↓
        FBRRateResolver
             ↓
        RateApplicabilityResolver
             ↓
        FBRRateChangeDetector
             ↓
        InvoiceRateResolution

    Important:

        A retrieved percentage is NOT automatically the
        applicable invoice rate.

    For example:

        18% = standard/base sales tax
         4% = further tax

    The 4% further tax must not replace the 18%
    base sales-tax rate.

    Classification is performed per extracted rate,
    not per retrieved chunk.
    """

    def __init__(
        self,
        current_rate_service,
    ):
        """
        Initialize the invoice-rate resolver.
        """

        if current_rate_service is None:
            raise ValueError(
                "current_rate_service cannot be None"
            )

        self.current_rate_service = (
            current_rate_service
        )

        self.rate_change_detector = (
            FBRRateChangeDetector()
        )

        self.applicability_resolver = (
            FBRRateApplicabilityResolver()
        )

    # ========================================================
    # BUILD QUERY
    # ========================================================

    @staticmethod
    def build_query(
        item_description: str,
        hs_code: str | None = None,
        invoice_date: str | None = None,
        purchase_type: str | None = None,
        invoice_type: str | None = None,
    ) -> str:
        """
        Build an invoice-aware FBR rate query.
        """

        invoice_query = InvoiceRateQuery(
            item_description=item_description,
            hs_code=hs_code,
            invoice_date=invoice_date,
            purchase_type=purchase_type,
            invoice_type=invoice_type,
        )

        return invoice_query.build_query()

    # ========================================================
    # RATE-CHANGE DETECTION
    # ========================================================

    def detect_rate_changes(
        self,
        results: list[dict[str, Any]],
    ) -> list[Any]:
        """
        Detect possible sales-tax changes from the same
        FBR evidence returned by retrieve().

        This is evidence detection only.

        It does not decide whether a rate change applies
        to the invoice.
        """

        if not isinstance(
            results,
            list,
        ):
            return []

        if not results:
            return []

        return (
            self.rate_change_detector.detect(
                results
            )
        )

    # ========================================================
    # CANDIDATE CONVERSION
    # ========================================================

    @staticmethod
    def _candidate_to_dict(
        candidate: TaxRateCandidate,
    ) -> dict[str, Any]:
        """
        Convert TaxRateCandidate into the dictionary
        format expected by FBRRateApplicabilityResolver.
        """

        return {
            "rate": candidate.rate,

            "source": candidate.source,

            "page": candidate.page,

            "text": candidate.text,

            "authority_score": (
                candidate.authority_score
            ),

            "semantic_score": (
                candidate.semantic_score
            ),

            "retrieval_score": (
                candidate.retrieval_score
            ),

            "year": candidate.year,

            "category": candidate.category,

            "applicability": (
                candidate.applicability
            ),

            "context": candidate.context,

            "product_match_score": (
                candidate.product_match_score
            ),

            "hs_code_match": (
                candidate.hs_code_match
            ),

            "effective_from": (
                candidate.effective_from
            ),

            "effective_to": (
                candidate.effective_to
            ),

            "date_relevance_score": (
                candidate.date_relevance_score
            ),
        }

    # ========================================================
    # RATE-SPECIFIC CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_candidate(
        candidate: TaxRateCandidate,
    ) -> str:
        """
        Classify an individual extracted tax-rate candidate.

        Classification is performed against the specific
        occurrence of the extracted rate, not the entire
        retrieved FBR chunk.

        This prevents a chunk containing both 18% standard
        sales tax and 3%/4% further tax from classifying all
        rates as further tax.
        """

        text = (candidate.text or "").lower()
        rate = float(candidate.rate)

        if rate == 0.0:
            return "zero-rated"

        # ----------------------------------------------------
        # Locate the exact numeric rate occurrence
        # ----------------------------------------------------

        
        number_patterns = (
        rf"(?<!\d){rate:g}\s*%",
        rf"(?<!\d){rate:g}\s*(?:\[[^\]]*\]\s*)?per\s+cent",
        rf"(?<!\d){rate:g}\s*(?:\[[^\]]*\]\s*)?percent",)

        match = None

        for pattern in number_patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                break

        # ----------------------------------------------------
        # Locate English number representation
        # ----------------------------------------------------

        word_map = {
            1.0: "one",
            2.0: "two",
            3.0: "three",
            4.0: "four",
            5.0: "five",
            6.0: "six",
            7.0: "seven",
            8.0: "eight",
            9.0: "nine",
            10.0: "ten",
            11.0: "eleven",
            12.0: "twelve",
            13.0: "thirteen",
            14.0: "fourteen",
            15.0: "fifteen",
            16.0: "sixteen",
            17.0: "seventeen",
            18.0: "eighteen",
            19.0: "nineteen",
            20.0: "twenty",
            21.0: "twenty-one",
            22.0: "twenty-two",
            23.0: "twenty-three",
            24.0: "twenty-four",
            25.0: "twenty-five",
            30.0: "thirty",
            40.0: "forty",
            50.0: "fifty",
        }
        word = word_map.get(rate)

        if word:
            match = re.search(rf"\b{re.escape(word)}\b"
                                  rf"\s*(?:\[[^\]]*\]\s*)?"
                                  rf"(?:per\s*cent|percent)",
        text,
        flags=re.IGNORECASE,
    )

        if match is None:
            return "unknown"

        # ====================================================
        # LOCAL PROVISION CONTEXT
        # ====================================================

        # Use legal punctuation boundaries rather than a broad
        # character window. FBR PDF text often wraps a single
        # provision across lines, so newline is intentionally
        # not treated as a hard boundary here.

        previous_boundaries = [
            position
            for position in (
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
                text.rfind(":", 0, match.start()),
            )
            if position != -1
        ]

        start = (
            max(previous_boundaries)
            if previous_boundaries
            else -1
        )

        next_boundaries = [
            position
            for position in (
                text.find(".", match.end()),
                text.find(";", match.end()),
                text.find(":", match.end()),
            )
            if position != -1
        ]

        end = (
            min(next_boundaries)
            if next_boundaries
            else len(text)
        )

        context = text[start + 1:end]

        before = text[
            start + 1:match.start()
        ]

        after = text[
            match.end():end
        ]

        # ====================================================
        # FURTHER TAX
        # ====================================================

        # The phrase must be close to this specific rate.

        further_before = re.search(
            r"further\s+tax",
            before[-80:],
            flags=re.IGNORECASE,
        )

        further_after = re.search(
            r"further\s+tax",
            after[:80],
            flags=re.IGNORECASE,
        )

        if further_before or further_after:
            return "further"

        # ====================================================
        # ZERO-RATED
        # ====================================================

        zero_patterns = (
            r"zero[-\s]?rated",
            r"zero[-\s]?rating",
            r"zero\s+rate",
            r"zero\s+per\s+cent",
            r"zero\s+percent",
        )

        if any(
            re.search(
                pattern,
                context,
                flags=re.IGNORECASE,
            )
            for pattern in zero_patterns
        ):
            return "zero-rated"

        # ====================================================
        # EXEMPTION
        # ====================================================

        if (
            "exempt" in context
            or "exemption" in context
        ):
            return "exempt"

        # ====================================================
        # REDUCED RATE
        # ====================================================

        reduced_patterns = (
            r"reduced\s+rate",
            r"reduced\s+to",
            r"sales\s+tax\s+reduced",
            r"reduction\s+in\s+sales\s+tax",
        )

        if any(
            re.search(
                pattern,
                context,
                flags=re.IGNORECASE,
            )
            for pattern in reduced_patterns
        ):
            return "reduced"

        # ====================================================
        # ENHANCED RATE
        # ====================================================

        enhanced_patterns = (
            r"enhanced\s+rate",
            r"enhanced\s+sales\s+tax",
            r"enhanced\s+rate\s+of",
            r"increased\s+rate",
            r"sales\s+tax\s+increased",
            r"increase\s+in\s+sales\s+tax",
        )

        if any(
            re.search(
                pattern,
                context,
                flags=re.IGNORECASE,
            )
            for pattern in enhanced_patterns
        ):
            return "enhanced"

        # ====================================================
        # SPECIAL RATE
        # ====================================================

        if (
            "special rate" in context
            or "special sales tax" in context
        ):
            return "special"

        # ====================================================
        # STANDARD / BASE SALES TAX
        # ====================================================

        standard_patterns = (
            r"sales\s+tax\s+at\s+the\s+rate\s+of",
            r"sales\s+tax\s+at\s+a\s+rate\s+of",
            r"sales\s+tax\s+at\s+rate\s+of",
            r"tax\s+known\s+as\s+sales\s+tax",
            r"standard\s+rate",
            r"standard\s+sales\s+tax",
            r"taxable\s+supplies",
            r"scope\s+of\s+tax",
            r"goods\s+imported\s+into\s+pakistan",
            r"there\s+shall\s+be\s+charged",
            r"levied\s+and\s+paid\s+a\s+tax",
        )

        # ----------------------------------------------------
        # First check the local provision context.
        # ----------------------------------------------------

        if any(
            re.search(
                pattern,
                context,
                flags=re.IGNORECASE,
            )
            for pattern in standard_patterns
        ):
            return "standard"

        # ----------------------------------------------------
        # Wider rate-specific context
        # ----------------------------------------------------

        # This wider window is still rate-specific because it
        # is centered on the exact extracted rate occurrence.
        #
        # We do NOT classify the entire retrieved chunk.

        wide_start = max(
            0,
            match.start() - 300,
        )

        wide_end = min(
            len(text),
            match.end() + 300,
        )

        wide_context = text[
            wide_start:wide_end
        ]

        # ----------------------------------------------------
        # Further-tax association
        # ----------------------------------------------------

        # Look for "further tax" in the wider window.
        #
        # Crucially, the rate must also occur within the same
        # short phrase following "further tax".
        #
        # This prevents:
        #
        #     18% standard ... 3% further tax
        #
        # from classifying 18% as further merely because
        # "further tax" exists somewhere in the chunk.

        further_match = re.search(
            r"further\s+tax.{0,100}",
            wide_context,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if further_match:
            further_text = (
                further_match.group(0)
            )

            rate_words = (
                rf"\b{rate:g}\s*"
                rf"(?:%|percent|per\s+cent)\b"
            )

            word = word_map.get(rate)

            if word:
                rate_words += (
                    rf"|\b{re.escape(word)}\b"
                    rf"\s*(?:percent|per\s+cent)"
                )

            if re.search(
                rate_words,
                further_text,
                flags=re.IGNORECASE,
            ):
                return "further"

        # ----------------------------------------------------
        # Standard-rate fallback
        # ----------------------------------------------------

        if any(
            re.search(
                pattern,
                wide_context,
                flags=re.IGNORECASE,
            )
            for pattern in standard_patterns
        ):
            return "standard"

        return "unknown"

    # ========================================================
    # RATE-SPECIFIC CLASSIFICATION ALIAS
    # ========================================================

    @staticmethod
    def _classify_rate_candidate(
        candidate: TaxRateCandidate,
    ) -> str:
        """Backward-compatible alias for rate classification."""

        return FBRInvoiceRateResolver._classify_candidate(
            candidate
        )

    # ========================================================
    # DATE RELEVANCE
    # ========================================================

    @staticmethod
    def _date_relevance(
        effective_from: str | None,
        invoice_date: str | None,
    ) -> float:
        """
        Calculate whether a candidate document is
        temporally relevant to the invoice date.

        Returns:

            1.0 for a document effective on/before
            the invoice date.

            0.0 for a future document.

            0.0 when either date is unavailable or invalid.

        Important:

            This is a temporal retrieval signal.

            It is NOT legal proof that the rate was
            continuously effective throughout the period.
        """

        if not invoice_date:
            return 0.0

        if not effective_from:
            return 0.0

        try:
            invoice = date.fromisoformat(
                invoice_date
            )

            effective = date.fromisoformat(
                effective_from
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        # ----------------------------------------------------
        # Future document
        # ----------------------------------------------------

        if effective > invoice:
            return 0.0

        # ----------------------------------------------------
        # Document is on/before invoice date
        # ----------------------------------------------------

        return 1.0

    # ========================================================
    # RATE-SPECIFIC CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_rate_in_context(
        rate: float,
        text: str,
    ) -> str:
        """
        Classify one extracted rate using the local legal
        context associated with that specific rate.

        A retrieved FBR chunk may contain multiple provisions,
        for example:

            18% standard sales tax
             3% further tax
             0% zero-rated

        Classification therefore operates on the individual
        rate occurrence rather than on the complete chunk.
        """

        if not isinstance(
            text,
            str,
        ) or not text.strip():
            return "unknown"

        text_lower = text.lower()

        if float(rate) == 0.0:
            return "zero-rated"

        positions: list[
            tuple[int, int]
        ] = []

        numeric_patterns = (
            rf"(?<![\d.])"
            rf"{re.escape(f'{rate:g}')}"
            rf"\s*%",

            rf"(?<![\d.])"
            rf"{re.escape(f'{rate:g}')}"
            rf"\s+per\s+cent\b",

            rf"(?<![\d.])"
            rf"{re.escape(f'{rate:g}')}"
            rf"\s+percent\b",
        )

        for pattern in numeric_patterns:
            for match in re.finditer(
                pattern,
                text_lower,
                flags=re.IGNORECASE,
            ):
                positions.append(
                    (
                        match.start(),
                        match.end(),
                    )
                )

        word_map = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty",
            21: "twenty-one",
            22: "twenty-two",
            23: "twenty-three",
            24: "twenty-four",
            25: "twenty-five",
            30: "thirty",
            40: "forty",
            50: "fifty",
        }

        if float(rate).is_integer():
            word = word_map.get(
                int(rate)
            )

            if word:
                word_pattern = (
                    rf"\b{re.escape(word)}\b"
                    rf"\s+(?:per\s+cent|percent)\b"
                )

                for match in re.finditer(
                    word_pattern,
                    text_lower,
                    flags=re.IGNORECASE,
                ):
                    positions.append(
                        (
                            match.start(),
                            match.end(),
                        )
                    )

        if not positions:
            return "unknown"

        for start, end in positions:

            previous_boundaries = [
                position
                for position in (
                    text_lower.rfind(
                        ".",
                        0,
                        start,
                    ),
                    text_lower.rfind(
                        ";",
                        0,
                        start,
                    ),
                    text_lower.rfind(
                        ":",
                        0,
                        start,
                    ),
                )
                if position != -1
            ]

            provision_start = (
                max(previous_boundaries)
                if previous_boundaries
                else -1
            )

            next_boundaries = [
                position
                for position in (
                    text_lower.find(
                        ".",
                        end,
                    ),
                    text_lower.find(
                        ";",
                        end,
                    ),
                    text_lower.find(
                        ":",
                        end,
                    ),
                )
                if position != -1
            ]

            provision_end = (
                min(next_boundaries)
                if next_boundaries
                else len(text_lower)
            )

            context = text_lower[
                provision_start + 1:
                provision_end
            ]

            before = text_lower[
                provision_start + 1:
                start
            ]

            after = text_lower[
                end:
                provision_end
            ]

            # ------------------------------------------------
            # Further tax
            # ------------------------------------------------

            if (
                re.search(
                    r"further\s+tax",
                    before[-80:],
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"further\s+tax",
                    after[:80],
                    flags=re.IGNORECASE,
                )
            ):
                return "further"

            # ------------------------------------------------
            # Zero-rated
            # ------------------------------------------------

            if (
                "zero-rated" in context
                or "zero rated" in context
                or "zero rating" in context
                or "zero per cent" in context
                or "zero percent" in context
            ):
                return "zero-rated"

            # ------------------------------------------------
            # Exemption
            # ------------------------------------------------

            if (
                "exempt" in context
                or "exemption" in context
            ):
                return "exempt"

            # ------------------------------------------------
            # Reduced
            # ------------------------------------------------

            if (
                "reduced rate" in context
                or "reduced to" in context
                or "sales tax reduced" in context
            ):
                return "reduced"

            # ------------------------------------------------
            # Enhanced
            # ------------------------------------------------

            if (
                "enhanced rate" in context
                or "enhanced sales tax" in context
                or "enhanced rate of" in context
            ):
                return "enhanced"

            # ------------------------------------------------
            # Special
            # ------------------------------------------------

            if (
                "special rate" in context
                or "special sales tax" in context
            ):
                return "special"

            # ------------------------------------------------
            # Standard/base
            # ------------------------------------------------

            standard_indicators = (
                "scope of tax",
                "sales tax at the rate of",
                "sales tax at rate of",
                "taxable supplies",
                "goods imported into pakistan",
                "there shall be charged, levied and paid a tax",
                "tax known as sales tax at the rate",
                "standard sales tax",
                "standard rate",
            )

            if any(
                indicator in context
                for indicator in standard_indicators
            ):
                return "standard"

        return "unknown"

    # ========================================================
    # BUILD APPLICABILITY CANDIDATES
    # ========================================================

    def _build_applicability_candidates(
        self,
        results: list[dict[str, Any]],
        invoice_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert retrieved FBR evidence into candidates
        for the applicability resolver.

        Each extracted rate is classified independently.

        Example:

            18% standard
             4% further
             0% zero-rated

        The three rates remain separate candidates.

        Date relevance is calculated using the document's
        cutoff/amendment date and the invoice date.
        """

        candidates: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # PROCESS RETRIEVED RESULTS
        # ====================================================

        for result in results:

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

            page = metadata.get(
                "page",
                None,
            )

            text = result.get(
                "text",
                "",
            )

            # ------------------------------------------------
            # Rate resolver
            # ------------------------------------------------

            resolver = (
                self.current_rate_service.rate_resolver
            )

            # ------------------------------------------------
            # Extract rates
            # ------------------------------------------------

            rates = resolver.extract_rates(
                text
            )

            if not rates:
                continue

            # =================================================
            # SCORES
            # =================================================

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

            # =================================================
            # YEAR
            # =================================================

            year = resolver.extract_year(
                source
            )

            # =================================================
            # CUTOFF DATE
            # =================================================

            effective_from = None

            if hasattr(
                resolver,
                "extract_cutoff_date",
            ):
                effective_from = (
                    resolver.extract_cutoff_date(
                        source
                    )
                )

            # =================================================
            # DATE RELEVANCE
            # =================================================

            date_relevance_score = (
                self._date_relevance(
                    effective_from=effective_from,
                    invoice_date=invoice_date,
                )
            )

            # =================================================
            # CREATE ONE CANDIDATE PER RATE
            # =================================================

            for rate in rates:

                # ------------------------------------------------
                # Create TaxRateCandidate
                # ------------------------------------------------

                # Construct the TaxRateCandidate FIRST.
                #
                # Classification must operate on the exact
                # candidate/rate occurrence rather than treating
                # the whole retrieved chunk as one rate.

                candidate = TaxRateCandidate(
                    rate=rate,
                    source=source,
                    page=page,
                    text=text,
                    authority_score=authority_score,
                    semantic_score=semantic_score,
                    retrieval_score=retrieval_score,
                    year=year,
                    effective_from=effective_from,
                    date_relevance_score=date_relevance_score,
                )

                # ------------------------------------------------
                # Reuse the canonical classifier used by the
                # direct regression tests.
                # ------------------------------------------------

                category = self._classify_candidate(
                    candidate
                )
                print("DEBUG CLASSIFICATION:","rate=",
    rate,
    "category=",
    category,
)

                # ------------------------------------------------
                # Zero-rate safety
                # ------------------------------------------------

                if rate == 0.0:
                    category = "zero-rated"

                # ------------------------------------------------
                # Applicability mapping
                # ------------------------------------------------

                if category == "standard":
                    applicability = "base"

                elif category == "further":
                    applicability = "conditional"

                elif category in (
                    "reduced",
                    "enhanced",
                    "special",
                    "zero-rated",
                    "exempt",
                ):
                    applicability = "conditional"

                else:
                    applicability = "unknown"

                # =================================================
                # STORE CANDIDATE
                # =================================================

                candidates.append(
                    {
                        "rate": rate,

                        "category": category,

                        "applicability": (
                            applicability
                        ),

                        "source": source,

                        "page": page,

                        "text": text,

                        "authority_score": (
                            authority_score
                        ),

                        "semantic_score": (
                            semantic_score
                        ),

                        "retrieval_score": (
                            retrieval_score
                        ),

                        "year": year,

                        "confidence": (
                            retrieval_score
                        ),

                        "effective_from": (
                            effective_from
                        ),

                        "effective_to": None,

                        "date_relevance_score": (
                            date_relevance_score
                        ),
                    }
                )

        return candidates

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        item_description: str,
        hs_code: str | None = None,
        invoice_date: str | None = None,
        purchase_type: str | None = None,
        invoice_type: str | None = None,
    ) -> InvoiceRateResolution:
        """
        Resolve the applicable sales-tax rate for one
        invoice item.
        """

        # ====================================================
        # VALIDATE ITEM
        # ====================================================

        if not isinstance(
            item_description,
            str,
        ):
            raise TypeError(
                "item_description must be a string"
            )

        item_description = (
            item_description.strip()
        )

        if not item_description:
            raise ValueError(
                "item_description cannot be empty"
            )

        # ====================================================
        # BUILD QUERY
        # ====================================================

        query = self.build_query(
            item_description=item_description,
            hs_code=hs_code,
            invoice_date=invoice_date,
            purchase_type=purchase_type,
            invoice_type=invoice_type,
        )

        # ====================================================
        # RETRIEVE FBR EVIDENCE
        # ====================================================

        results = (
            self.current_rate_service.retrieve(
                query
            )
        )

        if not results:
            raise LookupError(
                "No FBR evidence was retrieved "
                "for the invoice item."
            )

        # ====================================================
        # DETECT RATE CHANGES
        # ====================================================

        change_candidates = (
            self.detect_rate_changes(
                results
            )
        )

        # ====================================================
        # BUILD RATE CANDIDATES
        # ====================================================

        applicability_candidates = (
            self._build_applicability_candidates(
                results,
                invoice_date=invoice_date,
            )
        )

        # ====================================================
        # TEMPORARY DEBUG: INVOICE APPLICABILITY CANDIDATES
        # ====================================================

        print()
        print("=" * 80)
        print("INVOICE APPLICABILITY CANDIDATES")
        print("=" * 80)

        for i, candidate in enumerate(
            applicability_candidates,
            start=1,
        ):
            print()
            print(f"Candidate #{i}")
            print(
                "Rate:",
                candidate.get("rate"),
            )
            print(
                "Category:",
                candidate.get("category"),
            )
            print(
                "Applicability:",
                candidate.get("applicability"),
            )
            print(
                "Year:",
                candidate.get("year"),
            )
            print(
                "Effective from:",
                candidate.get(
                    "effective_from"
                ),
            )
            print(
                "Effective to:",
                candidate.get(
                    "effective_to"
                ),
            )
            print(
                "Date relevance:",
                candidate.get(
                    "date_relevance_score"
                ),
            )
            print(
                "Retrieval:",
                candidate.get(
                    "retrieval_score"
                ),
            )
            print(
                "Authority:",
                candidate.get(
                    "authority_score"
                ),
            )
            print("=" * 80)

        if not applicability_candidates:
            raise LookupError(
                "No usable sales-tax rate candidates "
                "were extracted from the FBR evidence."
            )

        # ====================================================
        # DETERMINE APPLICABILITY
        # ====================================================

        applicable = (
            self.applicability_resolver.resolve(
                applicability_candidates
            )
        )

        if applicable is None:
            raise LookupError(
                "The applicability resolver returned "
                "no result."
            )

        if applicable.base_rate is None:
            raise LookupError(
                "No applicable base sales-tax rate "
                "could be determined from the retrieved "
                "FBR evidence."
            )

        # ====================================================
        # FIND MATCHING SOURCE
        # ====================================================

        selected_candidate = None

        for result in applicability_candidates:

            if (
                result["rate"]
                == applicable.base_rate
            ):
                selected_candidate = result
                break

        # ====================================================
        # BUILD TAX RATE CANDIDATE
        # ====================================================

        selected_tax_candidate = None

        if selected_candidate is not None:

            selected_tax_candidate = (
                TaxRateCandidate(
                    rate=selected_candidate[
                        "rate"
                    ],

                    source=selected_candidate.get(
                        "source",
                        "",
                    ),

                    page=selected_candidate.get(
                        "page"
                    ),

                    text=selected_candidate.get(
                        "text",
                        "",
                    ),

                    authority_score=(
                        selected_candidate.get(
                            "authority_score",
                            0.0,
                        )
                    ),

                    semantic_score=(
                        selected_candidate.get(
                            "semantic_score",
                            0.0,
                        )
                    ),

                    retrieval_score=(
                        selected_candidate.get(
                            "retrieval_score",
                            0.0,
                        )
                    ),

                    year=selected_candidate.get(
                        "year"
                    ),

                    category=selected_candidate.get(
                        "category",
                        "unknown",
                    ),

                    applicability=selected_candidate.get(
                        "applicability",
                        "unknown",
                    ),

                    context=selected_candidate.get(
                        "text",
                        "",
                    ),

                    effective_from=(
                        selected_candidate.get(
                            "effective_from"
                        )
                    ),

                    effective_to=(
                        selected_candidate.get(
                            "effective_to"
                        )
                    ),

                    date_relevance_score=(
                        selected_candidate.get(
                            "date_relevance_score",
                            0.0,
                        )
                    ),
                )
            )

        # ====================================================
        # FINAL RESOLUTION
        # ====================================================

        return InvoiceRateResolution(
            item_description=item_description,

            hs_code=hs_code,

            invoice_date=invoice_date,

            rate=applicable.base_rate,

            category=applicable.category,

            confidence=applicable.confidence,

            source=applicable.source,

            page=applicable.page,

            text=(
                selected_candidate.get(
                    "text",
                    "",
                )
                if selected_candidate
                else ""
            ),

            query=query,

            candidate=selected_tax_candidate,

            change_candidates=(
                change_candidates
            ),

            additional_rate=(
                applicable.additional_rate
            ),

            applicability_explanation=(
                applicable.explanation
            ),
        )