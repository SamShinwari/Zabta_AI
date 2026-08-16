from dataclasses import dataclass
from pathlib import Path
import json

import faiss
import numpy as np

from src.fbr.chunker import FBRChunk


# ============================================================
# Vector Record
# ============================================================

@dataclass
class FBRVectorRecord:
    """
    Metadata connecting a FAISS vector to its original FBR chunk.
    """

    vector_id: int
    chunk_id: str
    text: str
    source: str
    category: str
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict

    def to_dict(self) -> dict:
        """
        Convert record to a JSON-serializable dictionary.
        """

        return {
            "vector_id": self.vector_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


# ============================================================
# FBR Vector Store
# ============================================================

class FBRVectorStore:
    """
    FAISS vector store for FBR document chunks.

    Uses IndexFlatIP because embeddings are normalized.
    Inner product therefore behaves like cosine similarity.
    """

    def __init__(
        self,
        dimension: int,
    ):
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero"
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.records: list[FBRVectorRecord] = []

    # --------------------------------------------------------
    # Add Vectors
    # --------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[FBRChunk],
    ) -> None:
        """
        Add embeddings and their corresponding chunks.
        """

        if len(chunks) == 0:
            return

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a 2D array"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of embeddings must match "
                "number of chunks"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension does not match "
                "vector store dimension"
            )

        start_id = len(self.records)

        self.index.add(
            embeddings
        )

        for offset, chunk in enumerate(chunks):

            vector_id = start_id + offset

            self.records.append(
                FBRVectorRecord(
                    vector_id=vector_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    category=chunk.category,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                )
            )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[FBRVectorRecord, float]]:
        """
        Search the vector store.

        Returns:
            List of (record, similarity_score)
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1, -1
            )

        if query_embedding.ndim != 2:
            raise ValueError(
                "query_embedding must be a 1D or 2D array"
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension does not match "
                "vector store dimension"
            )

        actual_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            actual_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            results.append(
                (
                    self.records[int(index)],
                    float(score),
                )
            )

        return results

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of vectors in the store.
        """

        return int(
            self.index.ntotal
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Save FAISS index and metadata.
        """

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            directory / "fbr.index"
        )

        metadata_path = (
            directory / "metadata.json"
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata = {
            "dimension": self.dimension,
            "count": self.size,
            "records": [
                record.to_dict()
                for record in self.records
            ],
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FBRVectorStore":
        """
        Load a previously saved vector store.
        """

        directory = Path(directory)

        index_path = (
            directory / "fbr.index"
        )

        metadata_path = (
            directory / "metadata.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        store = cls(
            dimension=int(
                metadata["dimension"]
            )
        )

        store.index = index

        store.records = [
            FBRVectorRecord(
                vector_id=int(
                    record["vector_id"]
                ),
                chunk_id=record["chunk_id"],
                text=record["text"],
                source=record["source"],
                category=record["category"],
                page_start=int(
                    record["page_start"]
                ),
                page_end=int(
                    record["page_end"]
                ),
                chunk_index=int(
                    record["chunk_index"]
                ),
                metadata=record.get(
                    "metadata",
                    {},
                ),
            )
            for record in metadata["records"]
        ]

        return store
