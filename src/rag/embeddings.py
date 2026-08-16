from sentence_transformers import SentenceTransformer


# Same embedding model used to create the existing
# FBR vector database.
EMBEDDING_MODEL = "BAAI/bge-m3"


class FBREmbeddings:
    """
    Generates embeddings for FBR RAG queries.

    The query embedding dimension must match
    the existing FAISS index dimension.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def embed_query(self, query: str):
        """
        Generate a normalized embedding for one query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding
