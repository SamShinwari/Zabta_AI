import numpy as np
import pytest

from src.fbr.chunker import FBRChunk
from src.fbr.embeddings import (
    FBRChunk,
    FBREmbedding,
    FBREmbeddingModel,
)


# ============================================================
# Helpers
# ============================================================

def make_chunk(
    chunk_id="chunk_001",
    text="Federal Board of Revenue sales tax regulations.",
):
    return FBRChunk(
        chunk_id=chunk_id,
        text=text,
        source="test.pdf",
        category="sro",
        page_start=1,
        page_end=1,
        chunk_index=0,
        char_count=len(text),
        metadata={
            "source": "test.pdf",
            "category": "sro",
            "page_start": 1,
            "page_end": 1,
        },
    )


# ============================================================
# Model
# ============================================================

def test_model_loads():

    model = FBREmbeddingModel()

    assert model.model is not None


def test_dimension():

    model = FBREmbeddingModel()

    assert model.dimension > 0


# ============================================================
# Single Text
# ============================================================

def test_embed_text():

    model = FBREmbeddingModel()

    vector = model.embed_text(
        "Federal Board of Revenue"
    )

    assert isinstance(
        vector,
        np.ndarray,
    )

    assert vector.dtype == np.float32

    assert vector.ndim == 1

    assert len(vector) == model.dimension


def test_empty_text_rejected():

    model = FBREmbeddingModel()

    with pytest.raises(ValueError):

        model.embed_text("")


# ============================================================
# Multiple Texts
# ============================================================

def test_embed_texts():

    model = FBREmbeddingModel()

    vectors = model.embed_texts(
        [
            "Sales tax",
            "Income tax",
        ]
    )

    assert isinstance(
        vectors,
        np.ndarray,
    )

    assert vectors.ndim == 2

    assert vectors.shape[0] == 2

    assert vectors.shape[1] == model.dimension


def test_empty_texts():

    model = FBREmbeddingModel()

    vectors = model.embed_texts([])

    assert isinstance(
        vectors,
        np.ndarray,
    )

    assert vectors.shape == (0, 0)


# ============================================================
# Chunk Embeddings
# ============================================================

def test_embed_chunks():

    model = FBREmbeddingModel()

    chunks = [
        make_chunk("chunk_001"),
        make_chunk(
            "chunk_002",
            "Sales tax registration requirements.",
        ),
    ]

    results = model.embed_chunks(
        chunks
    )

    assert len(results) == 2

    assert isinstance(
        results[0],
        FBREmbedding,
    )

    assert results[0].chunk_id == "chunk_001"

    assert isinstance(
        results[0].embedding,
        np.ndarray,
    )

    assert len(
        results[0].embedding
    ) == model.dimension


def test_empty_chunks():

    model = FBREmbeddingModel()

    results = model.embed_chunks([])

    assert results == []


# ============================================================
# Matrix
# ============================================================

def test_chunks_to_matrix():

    model = FBREmbeddingModel()

    chunks = [
        make_chunk("chunk_001"),
        make_chunk(
            "chunk_002",
            "Sales tax registration requirements.",
        ),
        make_chunk(
            "chunk_003",
            "Federal excise duty regulations.",
        ),
    ]

    matrix = model.chunks_to_matrix(
        chunks
    )

    assert isinstance(
        matrix,
        np.ndarray,
    )

    assert matrix.ndim == 2

    assert matrix.shape == (
        3,
        model.dimension,
    )

    assert matrix.dtype == np.float32


def test_empty_matrix():

    model = FBREmbeddingModel()

    matrix = model.chunks_to_matrix([])

    assert matrix.shape == (0, 0)


# ============================================================
# Serialization
# ============================================================

def test_embedding_to_dict():

    embedding = FBREmbedding(
        chunk_id="chunk_001",
        embedding=np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
    )

    result = embedding.to_dict()

    assert result["chunk_id"] == "chunk_001"

    assert isinstance(
        result["embedding"],
        list,
    )

    assert len(
        result["embedding"]
    ) == 3
