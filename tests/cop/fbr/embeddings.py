from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.fbr.chunker import FBRChunk


# ============================================================
# Embedding Result
# ============================================================

@dataclass
class FBREmbedding:
    """
    Embedding generated for one FBR document chunk.
    """

    chunk_id: str
    embedding: np.ndarray

    def to_dict(self) -> dict:
        """
        Convert embedding information to a dictionary.

        The vector itself is converted to a Python list so that
        the result can be serialized if required.
        """

        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding.tolist(),
        }


# ============================================================
# FBR Embedding Model
# ============================================================

class FBREmbeddingModel:
    """
    Generate embeddings for FBR document chunks.

    Default model:
        BAAI/bge-m3
    """

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name
        )

    # --------------------------------------------------------
    # Single Text
    # --------------------------------------------------------

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate an embedding for one text string.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Cannot embed empty text"
            )

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Multiple Texts
    # --------------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        cleaned_texts = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "Every text must be a string"
                )

            text = text.strip()

            if not text:
                raise ValueError(
                    "Cannot embed empty text"
                )

            cleaned_texts.append(text)

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # FBR Chunks
    # --------------------------------------------------------

    def embed_chunks(
        self,
        chunks: Sequence[FBRChunk],
        batch_size: int = 32,
    ) -> list[FBREmbedding]:
        """
        Generate embeddings for FBR chunks.
        """

        if not chunks:
            return []

        texts = [
            chunk.text
            for chunk in chunks
        ]

        vectors = self.embed_texts(
            texts,
            batch_size=batch_size,
        )

        results = []

        for chunk, vector in zip(
            chunks,
            vectors,
        ):

            results.append(
                FBREmbedding(
                    chunk_id=chunk.chunk_id,
                    embedding=vector,
                )
            )

        return results

    # --------------------------------------------------------
    # Vector Matrix
    # --------------------------------------------------------

    def chunks_to_matrix(
        self,
        chunks: Sequence[FBRChunk],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate a 2D embedding matrix for FAISS.

        Shape:

            (number_of_chunks, embedding_dimension)
        """

        if not chunks:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        return self.embed_texts(
            texts,
            batch_size=batch_size,
        )

    # --------------------------------------------------------
    # Embedding Dimension
    # --------------------------------------------------------

    @property
    def dimension(self) -> int:
        """
        Return the embedding vector dimension.
        """

        return int(
            self.model.get_embedding_dimension()
        )
