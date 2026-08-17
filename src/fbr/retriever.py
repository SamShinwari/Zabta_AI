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
        BGE-M3 Embedding
             ↓
        FAISS Semantic Search
             ↓
        Legal Reference Matching
             ↓
        Legal Reference Boost
             ↓
        Final Retrieval Results
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

            section = str(section).strip().lower()

            if not section:
                continue

            # Example:
            #
            # Section 8
            # Section 8B
            # section 8B.
            #

            pattern = (
                rf"\bsection\s+"
                rf"{re.escape(section)}\b"
            )

            if re.search(
                pattern,
                combined_text,
            ):
                return True

            # ------------------------------------------------
            # Handle extracted PDF formats:
            #
            # 8B.
            # 8B-
            # 8B:
            # ------------------------------------------------

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

            rule = str(rule).strip().lower()

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

            # Handle formats such as:
            #
            # Rule 12.
            # Rule 12-
            #

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

            sro = str(sro).strip().lower()

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

        Pipeline:

            1. Validate query
            2. Analyze query
            3. Generate BGE-M3 embedding
            4. Search FAISS
            5. Match legal references
            6. Apply legal-reference boost
            7. Re-rank boosted results
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

        analysis = self.query_analyzer.analyze(
            query
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
        ).reshape(1, -1)

        # ----------------------------------------------------
        # Validate embedding dimension
        # ----------------------------------------------------

        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not "
                "match FAISS index: "
                f"query={query_embedding.shape[1]}, "
                f"index={self.index.d}"
            )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        actual_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            actual_k,
        )

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results: list[dict[str, Any]] = []

        for rank, (score, index_id) in enumerate(
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
                "rank": rank,
                "score": float(score),
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

        # ----------------------------------------------------
        # Legal-reference boost
        # ----------------------------------------------------

        if analysis.has_legal_reference():

            for result in results:

                matched = self._legal_reference_match(
                    result=result,
                    sections=analysis.sections,
                    rules=analysis.rules,
                    sros=analysis.sros,
                )

                result[
                    "legal_reference_match"
                ] = matched

                if matched:

                    result[
                        "original_score"
                    ] = result["score"]

                    result[
                        "score"
                    ] += 0.10

        # ----------------------------------------------------
        # Re-sort results
        # ----------------------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Reassign ranks
        # ----------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result["rank"] = rank

        # ----------------------------------------------------
        # Add query analysis information
        # ----------------------------------------------------

        for result in results:

            result[
                "query_analysis"
            ] = {
                "sections": analysis.sections,
                "rules": analysis.rules,
                "sros": analysis.sros,
                "has_legal_reference": (
                    analysis.has_legal_reference()
                ),
            }

        return results

    # ========================================================
    # STATISTICS
    # ========================================================

    @property
    def vector_count(self) -> int:
        """
        Number of vectors stored in FAISS.
        """

        return int(
            self.index.ntotal
        )

    @property
    def dimension(self) -> int:
        """
        Embedding/vector dimension.
        """

        return int(
            self.index.d
        )