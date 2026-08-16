from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.fbr.embeddings import FBREmbeddingModel


class FBRRetriever:
    """
    Retriever for the generated FBR FAISS vector database.

    Vector store format:

        data/vector_database/fbr/
            index.faiss
            metadata.json
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

            self.metadata: list[dict[str, Any]] = json.load(file)

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "FAISS index and metadata count do not match: "
                f"index={self.index.ntotal}, "
                f"metadata={len(self.metadata)}"
            )

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

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search FBR documents using semantic similarity.
        """

        if not isinstance(query, str):
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
        # Embed query
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

        results = []

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
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

            results.append(result)

        return results

    # ========================================================
    # Statistics
    # ========================================================

    @property
    def vector_count(self) -> int:
        return int(
            self.index.ntotal
        )

    @property
    def dimension(self) -> int:
        return int(
            self.index.d
        )

