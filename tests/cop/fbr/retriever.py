from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.fbr.embeddings import FBREmbeddingModel
from src.fbr.query_analyzer import FBRQueryAnalyzer


class FBRRetriever:
    """
    Retriever for the generated FBR FAISS vector database.

    Vector store format:

        data/vector_database/fbr/
            index.faiss
            metadata.json

    Retrieval pipeline:

        User Query
             ↓
        Query Analyzer
             ↓
        Query Type Detection
             ↓
        Candidate Size Selection
             ↓
        BGE-M3 Embedding
             ↓
        FAISS Semantic Search
             ↓
        Legal Reference Matching
             ↓
        Legal Reference Boost
             ↓
        Authority Scoring
             ↓
        Authority-Aware Ranking
             ↓
        Final Top-K Results
    """

    def __init__(
        self,
        vector_dir: str | Path,
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.vector_dir = Path(vector_dir)

        self.index_path = (
            self.vector_dir / "index.faiss"
        )

        self.metadata_path = (
            self.vector_dir / "metadata.json"
        )

        # ----------------------------------------------------
        # Validate vector database
        # ----------------------------------------------------

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        self.index = faiss.read_index(
            str(self.index_path)
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata: list[
                dict[str, Any]
            ] = json.load(file)

        # ----------------------------------------------------
        # Validate FAISS / metadata
        # ----------------------------------------------------

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "FAISS index and metadata count do not match: "
                f"index={self.index.ntotal}, "
                f"metadata={len(self.metadata)}"
            )

        # ----------------------------------------------------
        # Validate embedding dimension
        # ----------------------------------------------------

        if self.index.d != 1024:
            raise ValueError(
                f"Expected FAISS dimension 1024, "
                f"got {self.index.d}"
            )

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        self.embedding_model = FBREmbeddingModel(
            model_name=embedding_model,
            normalize_embeddings=True,
        )

        # ----------------------------------------------------
        # Query analyzer
        # ----------------------------------------------------

        self.query_analyzer = FBRQueryAnalyzer()

    # ========================================================
    # AUTHORITY SCORING
    # ========================================================

    @staticmethod
    def _authority_score(
        source: str,
        text: str = "",
        query_type: str = "",
    ) -> float:
        """
        Calculate document authority for retrieval.

        Authority depends on:
            - document type
            - tax domain
            - query type

        For sales-tax rate queries, Sales Tax Act,
        Finance Act, relevant SROs and notifications
        receive higher priority.
        """

        source_lower = source.lower()
        text_lower = text.lower()
        query_type_lower = query_type.lower()

        combined = (
            f"{source_lower}\n{text_lower}"
        )

        # ----------------------------------------------------
        # SALES TAX RATE QUERY
        # ----------------------------------------------------

        if (
            query_type_lower == "rate"
            or "sales tax rate" in query_type_lower
        ):

            # ------------------------------------------------
            # SALES TAX ACT
            # ------------------------------------------------

            if "sales tax act" in combined:
                return 1.00

            # ------------------------------------------------
            # FINANCE ACT
            # ------------------------------------------------

            if "finance act" in combined:
                return 0.95

            # ------------------------------------------------
            # SALES TAX SRO
            # ------------------------------------------------

            if (
                (
                    "s.r.o." in combined
                    or "sro" in combined
                )
                and "sales tax" in combined
            ):
                return 0.90

            # ------------------------------------------------
            # SALES TAX RULES
            # ------------------------------------------------

            if (
                "sales tax rules" in combined
                or "sales tax rule" in combined
            ):
                return 0.88

            # ------------------------------------------------
            # SALES TAX NOTIFICATION
            # ------------------------------------------------

            if (
                "notification" in combined
                and "sales tax" in combined
            ):
                return 0.82

            # ------------------------------------------------
            # SALES TAX CIRCULAR
            # ------------------------------------------------

            if (
                "circular" in combined
                and "sales tax" in combined
            ):
                return 0.75

            # ------------------------------------------------
            # INCOME TAX DOCUMENTS
            # ------------------------------------------------

            if (
                "income tax ordinance" in combined
                or "income tax act" in combined
                or "income tax rules" in combined
            ):
                return 0.20

            # ------------------------------------------------
            # TAX EXPENDITURE / REPORTS
            # ------------------------------------------------

            if (
                "tax expenditure" in combined
                or "report" in combined
                or "year book" in combined
            ):
                return 0.40

            # ------------------------------------------------
            # UNKNOWN
            # ------------------------------------------------

            return 0.30

        # ====================================================
        # NON-RATE QUERY
        # ====================================================

        # For non-rate questions, authority should have less
        # influence. We still provide a useful document
        # authority score for diagnostics.

        if "sales tax act" in combined:
            return 1.00

        if "finance act" in combined:
            return 0.95

        if (
            (
                "s.r.o." in combined
                or "sro" in combined
            )
            and "sales tax" in combined
        ):
            return 0.90

        if (
            "sales tax rules" in combined
            or "sales tax rule" in combined
        ):
            return 0.88

        if (
            "notification" in combined
            and "sales tax" in combined
        ):
            return 0.82

        if (
            "circular" in combined
            and "sales tax" in combined
        ):
            return 0.75

        if (
            "income tax ordinance" in combined
            or "income tax act" in combined
            or "income tax rules" in combined
        ):
            return 0.20

        if (
            "tax expenditure" in combined
            or "report" in combined
            or "year book" in combined
        ):
            return 0.40

        return 0.30

    # ========================================================
    # LEGAL REFERENCE MATCHING
    # ========================================================

    @staticmethod
    def _legal_reference_match(
        result: dict[str, Any],
        sections: list[str],
        rules: list[str],
        sros: list[str],
    ) -> bool:
        """
        Check whether a retrieved chunk explicitly contains
        the legal reference requested by the user.

        Supported references:

            - Section numbers
            - Rule numbers
            - SRO numbers
        """

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

        combined_text = (
            f"{source}\n{text}"
        ).lower()

        # ----------------------------------------------------
        # Section matching
        # ----------------------------------------------------

        for section in sections:

            section = str(
                section
            ).strip().lower()

            if not section:
                continue

            pattern = (
                rf"\bsection\s+"
                rf"{re.escape(section)}\b"
            )

            if re.search(
                pattern,
                combined_text,
            ):
                return True

            section_pattern = (
                rf"\b{re.escape(section)}"
                rf"\s*[\.\-\:]"
            )

            if re.search(
                section_pattern,
                combined_text,
            ):
                return True

        # ----------------------------------------------------
        # Rule matching
        # ----------------------------------------------------

        for rule in rules:

            rule = str(
                rule
            ).strip().lower()

            if not rule:
                continue

            pattern = (
                rf"\brule\s+"
                rf"{re.escape(rule)}\b"
            )

            if re.search(
                pattern,
                combined_text,
            ):
                return True

            rule_pattern = (
                rf"\brule\s+"
                rf"{re.escape(rule)}"
                rf"\s*[\.\-\:]"
            )

            if re.search(
                rule_pattern,
                combined_text,
            ):
                return True

        # ----------------------------------------------------
        # SRO matching
        # ----------------------------------------------------

        for sro in sros:

            sro = str(
                sro
            ).strip().lower()

            if not sro:
                continue

            if sro in combined_text:
                return True

        return False

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search FBR documents using semantic similarity.

        Candidate strategy:

            Normal query:
                candidate_k = top_k

            Rate query:
                candidate_k = max(50, top_k)

        The larger candidate pool for rate queries allows
        the system to consider older/newer Acts, Finance Acts,
        SROs, notifications and other legal sources before
        authority-aware ranking.

        Final output is still limited to top_k.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        # ----------------------------------------------------
        # Analyze query
        # ----------------------------------------------------

        analysis = (
            self.query_analyzer.analyze(
                query
            )
        )

        # ----------------------------------------------------
        # Determine query type
        # ----------------------------------------------------

        query_type = ""

        if hasattr(
            self.query_analyzer,
            "query_type",
        ):

            query_type = (
                self.query_analyzer.query_type(
                    query
                )
            )

        elif hasattr(
            analysis,
            "query_type",
        ):

            query_type = (
                analysis.query_type
            )

        # ----------------------------------------------------
        # Detect rate query
        # ----------------------------------------------------

        is_rate_query = (
            self.query_analyzer.is_rate_query(
                query
            )
        )

        # ====================================================
        # CANDIDATE SIZE
        # ====================================================

        if is_rate_query:

            # ------------------------------------------------
            # Rate queries require broader retrieval.
            #
            # We want to compare:
            #
            #   - current Sales Tax Act
            #   - Finance Act
            #   - SROs
            #   - Notifications
            #   - Rules
            #   - older versions
            #
            # before final ranking.
            # ------------------------------------------------

            candidate_k = max(
                50,
                top_k,
            )

        else:

            # ------------------------------------------------
            # Normal semantic retrieval
            # ------------------------------------------------

            candidate_k = top_k

        # Never ask FAISS for more vectors than exist.

        candidate_k = min(
            candidate_k,
            self.index.ntotal,
        )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_model.embed_text(
                query
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        # ----------------------------------------------------
        # Validate embedding dimension
        # ----------------------------------------------------

        if (
            query_embedding.shape[1]
            != self.index.d
        ):

            raise ValueError(
                "Query embedding dimension does not "
                "match FAISS index: "
                f"query={query_embedding.shape[1]}, "
                f"index={self.index.d}"
            )

        # ====================================================
        # FAISS CANDIDATE SEARCH
        # ====================================================

        scores, indices = (
            self.index.search(
                query_embedding,
                candidate_k,
            )
        )

        # ----------------------------------------------------
        # Build candidate results
        # ----------------------------------------------------

        results: list[
            dict[str, Any]
        ] = []

        for candidate_rank, (
            score,
            index_id,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):

            if index_id < 0:
                continue

            record = self.metadata[
                int(index_id)
            ]

            result = {
                # Candidate rank before
                # re-ranking.
                "candidate_rank": (
                    candidate_rank
                ),

                # Original FAISS score.
                "score": float(
                    score
                ),

                "text": record.get(
                    "text",
                    "",
                ),

                "metadata": record.get(
                    "metadata",
                    {},
                ),
            }

            results.append(
                result
            )

        # ====================================================
        # LEGAL REFERENCE BOOST
        # ====================================================

        if analysis.has_legal_reference():

            for result in results:

                matched = (
                    self._legal_reference_match(
                        result=result,
                        sections=(
                            analysis.sections
                        ),
                        rules=(
                            analysis.rules
                        ),
                        sros=(
                            analysis.sros
                        ),
                    )
                )

                result[
                    "legal_reference_match"
                ] = matched

                if matched:

                    # Preserve original FAISS
                    # score before boosting.

                    result[
                        "original_score"
                    ] = result[
                        "score"
                    ]

                    result[
                        "score"
                    ] += 0.10

        else:

            # Keep output structure consistent.

            for result in results:

                result[
                    "legal_reference_match"
                ] = False

        # ====================================================
        # AUTHORITY-AWARE RANKING
        # ====================================================

        if is_rate_query:

            for result in results:

                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                metadata = result.get(
                    "metadata",
                    {},
                )

                # ------------------------------------------------
                # Source
                # ------------------------------------------------

                source = metadata.get(
                    "source",
                    "",
                )

                # ------------------------------------------------
                # Authority
                # ------------------------------------------------

                authority = (
                    self._authority_score(
                        source=source,
                        text=result.get(
                            "text",
                            "",
                        ),
                        query_type=query_type
                        or "rate",
                    )
                )

                result[
                    "authority_score"
                ] = float(
                    authority
                )

                # ------------------------------------------------
                # Semantic score
                # ------------------------------------------------

                semantic_score = float(
                    result.get(
                        "score",
                        0.0,
                    )
                )

                # ------------------------------------------------
                # Retrieval score
                #
                # 80% semantic relevance
                # 20% authority
                # ------------------------------------------------

                result[
                    "retrieval_score"
                ] = (
                    0.80
                    * semantic_score
                    +
                    0.20
                    * authority
                )

        else:

            # ------------------------------------------------
            # Normal query
            #
            # Do not alter normal semantic ranking.
            # ------------------------------------------------

            for result in results:

                result[
                    "retrieval_score"
                ] = float(
                    result.get(
                        "score",
                        0.0,
                    )
                )

        # ====================================================
        # FINAL SORT
        # ====================================================

        results.sort(
            key=lambda item: item[
                "retrieval_score"
            ],
            reverse=True,
        )

        # ====================================================
        # FINAL TOP-K
        # ====================================================

        # We may have retrieved 50 candidates for a rate
        # query, but the caller still receives only top_k.

        results = results[
            :top_k
        ]

        # ====================================================
        # FINAL RANK
        # ====================================================

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result[
                "rank"
            ] = rank

        # ====================================================
        # QUERY ANALYSIS INFORMATION
        # ====================================================

        for result in results:

            result[
                "query_analysis"
            ] = {

                "sections": (
                    analysis.sections
                ),

                "rules": (
                    analysis.rules
                ),

                "sros": (
                    analysis.sros
                ),

                "has_legal_reference": (
                    analysis.has_legal_reference()
                ),

                "is_rate_query": (
                    is_rate_query
                ),

                "query_type": (
                    query_type
                ),

                "candidate_k": (
                    candidate_k
                ),

                "final_top_k": (
                    top_k
                ),
            }

        return results

    # ========================================================
    # STATISTICS
    # ========================================================

    @property
    def vector_count(
        self,
    ) -> int:
        """
        Number of vectors stored in FAISS.
        """

        return int(
            self.index.ntotal
        )

    @property
    def dimension(
        self,
    ) -> int:
        """
        Embedding/vector dimension.
        """

        return int(
            self.index.d
        )