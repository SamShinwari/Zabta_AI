from .embeddings import FBREmbeddings
from .vector_store import FBRVectorStore


class FBRRetriever:
    """
    High-level retriever for FBR documents.

    Combines:
        Query embedding
        +
        FAISS similarity search
        +
        FBR metadata
    """

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.top_k = top_k

        self.embeddings = FBREmbeddings()

        self.vector_store = FBRVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant FBR chunks.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        query_embedding = (
            self.embeddings.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=k,
        )

        return results
