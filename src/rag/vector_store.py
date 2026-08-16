import json
from pathlib import Path

import faiss


DEFAULT_INDEX_PATH = (
    "data/vector_database/fbr/index.faiss"
)

DEFAULT_METADATA_PATH = (
    "data/vector_database/fbr/metadata.json"
)


class FBRVectorStore:
    """
    Loads the existing FBR FAISS vector database
    and its metadata.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata size mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata records."
            )

    @property
    def dimension(self) -> int:
        return self.index.d

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):
        """
        Search the FAISS index.

        Returns:
            List of dictionaries containing score,
            text and metadata.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):
            if index_id < 0:
                continue

            record = self.metadata[index_id]

            results.append(
                {
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
            )

        return results
