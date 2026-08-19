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
        """
        Initialize the FBR current-rate service.
        """

        if retrieval_top_k <= 0:
            raise ValueError(
                "retrieval_top_k must be greater than zero"
            )

        self.retriever = FBRRetriever(
            vector_dir=vector_dir,
            embedding_model="BAAI/bge-m3",
        )

        self.query_analyzer = (
            FBRQueryAnalyzer()
        )

        self.rate_resolver = (
            FBRRateResolver()
        )

        self.retrieval_top_k = (
            retrieval_top_k
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve FBR evidence for a sales-tax question.

        This method exposes the retrieval layer so that
        downstream components such as the rate-change detector
        can inspect the same FBR evidence.
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

        results = self.retriever.search(
            question,
            top_k=self.retrieval_top_k,
        )

        if not results:
            raise LookupError(
                "No FBR evidence was retrieved "
                "for the sales tax query."
            )

        return results

    # ========================================================
    # STANDARD RATE RETRIEVAL
    # ========================================================

    def retrieve_standard_rate_evidence(
        self,
        invoice_date: str | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Retrieve evidence specifically for the
        base/standard sales-tax rate.

        This is intentionally different from the
        invoice/product query.

        The invoice query identifies product applicability.
        This query identifies the general statutory
        sales-tax rate.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query = (
            "Pakistan Sales Tax Act 1990 "
            "scope of tax "
            "standard sales tax rate "
            "taxable supplies "
            "goods imported into Pakistan "
            "there shall be charged levied and paid "
            "a tax known as sales tax at the rate of "
            "per cent"
        )

        return self.retriever.search(
            query,
            top_k=top_k,
        )

    # ========================================================
    # CURRENT STANDARD RATE
    # ========================================================

    def resolve_standard_rate(
        self,
        invoice_date: str | None = None,
        top_k: int = 20,
    ):
        """
        Resolve the strongest standard/base sales-tax
        rate from dedicated statutory-rate evidence.

        This method is intentionally separate from the
        generic invoice/product retrieval.

        Returns:
            TaxRateCandidate
        """

        results = (
            self.retrieve_standard_rate_evidence(
                invoice_date=invoice_date,
                top_k=top_k,
            )
        )

        if not results:
            raise LookupError(
                "No standard sales-tax evidence "
                "was retrieved."
            )

        resolved = (
            self.rate_resolver.resolve_standard_from_results(
                results,
                invoice_date=invoice_date,
            )
        )

        if resolved is None:
            raise LookupError(
                "No standard sales-tax rate "
                "could be resolved."
            )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # resolve_standard_from_results() returns:
        #
        # {
        #     "result": ...,
        #     "source": ...,
        #     "candidate": TaxRateCandidate,
        # }
        #
        # The public service method should return the
        # TaxRateCandidate itself because callers/tests
        # expect:
        #
        #     result.rate
        #     result.source
        #     result.page
        #     result.year
        # ----------------------------------------------------

        candidate = resolved.get(
            "candidate"
        )

        if candidate is None:
            raise LookupError(
                "Standard sales-tax resolution "
                "did not return a candidate."
            )

        return candidate

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        question: str,
        invoice_date: str | None = None,
    ) -> CurrentRateResult:
        """
        Resolve a sales-tax rate from FBR evidence.

        invoice_date is passed to the rate resolver so that
        candidate ranking can consider the relationship
        between the retrieved document cutoff date and
        the invoice date.

        Date relevance is a retrieval/ranking signal,
        NOT legal proof of rate applicability.
        """

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

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
        # Retrieve FBR evidence
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
        # Resolve best rate
        #
        # IMPORTANT:
        #
        # Pass invoice_date into resolve_from_results()
        # so date-aware ranking is actually used.
        # ----------------------------------------------------

        resolved = (
            self.rate_resolver.resolve_from_results(
                results,
                invoice_date=invoice_date,
            )
        )

        if resolved is None:
            raise LookupError(
                "No usable sales tax rate could be "
                "resolved from the retrieved FBR evidence."
            )

        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = resolved["result"]

        source = resolved.get(
            "source",
            {},
        )

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

    def info(
        self,
    ) -> dict[str, Any]:
        """
        Return configuration information.
        """

        return {
            "service": (
                "FBRCurrentRateService"
            ),

            "embedding_model": (
                "BAAI/bge-m3"
            ),

            "vector_directory": str(
                self.retriever.vector_dir
            ),

            "vector_count": (
                self.retriever.vector_count
            ),

            "embedding_dimension": (
                self.retriever.dimension
            ),

            "retrieval_top_k": (
                self.retrieval_top_k
            ),
        }