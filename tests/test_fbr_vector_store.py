import numpy as np
import pytest

from src.fbr.chunker import FBRChunk
from src.fbr.vector_store import (
    FBRVectorStore,
    FBRVectorRecord,
)


# ============================================================
# Helpers
# ============================================================

DIMENSION = 4


def make_chunk(
    chunk_id: str,
    text: str,
) -> FBRChunk:

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
# Initialization
# ============================================================

def test_store_initialization():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    assert store.dimension == DIMENSION
    assert store.size == 0
    assert store.index.ntotal == 0


def test_invalid_dimension():

    with pytest.raises(ValueError):

        FBRVectorStore(
            dimension=0
        )


# ============================================================
# Add
# ============================================================

def test_add_vectors():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            "chunk_001",
            "Sales tax regulations",
        ),
        make_chunk(
            "chunk_002",
            "Income tax regulations",
        ),
    ]

    store.add(
        embeddings,
        chunks,
    )

    assert store.size == 2

    assert len(
        store.records
    ) == 2


def test_add_empty():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    store.add(
        np.empty(
            (0, DIMENSION),
            dtype=np.float32,
        ),
        [],
    )

    assert store.size == 0


def test_mismatched_count():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.array(
        [[1, 0, 0, 0]],
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            "chunk_001",
            "Sales tax",
        ),
        make_chunk(
            "chunk_002",
            "Income tax",
        ),
    ]

    with pytest.raises(ValueError):

        store.add(
            embeddings,
            chunks,
        )


def test_wrong_dimension():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.array(
        [[1, 0, 0]],
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            "chunk_001",
            "Sales tax",
        )
    ]

    with pytest.raises(ValueError):

        store.add(
            embeddings,
            chunks,
        )


# ============================================================
# Search
# ============================================================

def test_search():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            "chunk_001",
            "Sales tax",
        ),
        make_chunk(
            "chunk_002",
            "Income tax",
        ),
        make_chunk(
            "chunk_003",
            "Federal excise duty",
        ),
    ]

    store.add(
        embeddings,
        chunks,
    )

    query = np.array(
        [1, 0, 0, 0],
        dtype=np.float32,
    )

    results = store.search(
        query,
        top_k=2,
    )

    assert len(results) == 2

    record, score = results[0]

    assert isinstance(
        record,
        FBRVectorRecord,
    )

    assert record.chunk_id == "chunk_001"

    assert score > 0.9


def test_search_empty_store():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    query = np.array(
        [1, 0, 0, 0],
        dtype=np.float32,
    )

    results = store.search(
        query
    )

    assert results == []


def test_search_top_k():

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.eye(
        DIMENSION,
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            f"chunk_{i}",
            f"Document {i}",
        )
        for i in range(DIMENSION)
    ]

    store.add(
        embeddings,
        chunks,
    )

    query = np.array(
        [1, 0, 0, 0],
        dtype=np.float32,
    )

    results = store.search(
        query,
        top_k=2,
    )

    assert len(results) == 2


# ============================================================
# Persistence
# ============================================================

def test_save_and_load(tmp_path):

    store = FBRVectorStore(
        dimension=DIMENSION
    )

    embeddings = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    chunks = [
        make_chunk(
            "chunk_001",
            "Sales tax regulations",
        ),
        make_chunk(
            "chunk_002",
            "Income tax regulations",
        ),
    ]

    store.add(
        embeddings,
        chunks,
    )

    store.save(
        tmp_path
    )

    loaded = FBRVectorStore.load(
        tmp_path
    )

    assert loaded.dimension == DIMENSION

    assert loaded.size == 2

    assert len(
        loaded.records
    ) == 2

    query = np.array(
        [1, 0, 0, 0],
        dtype=np.float32,
    )

    results = loaded.search(
        query,
        top_k=1,
    )

    assert results[0][0].chunk_id == "chunk_001"


def test_missing_index(tmp_path):

    with pytest.raises(
        FileNotFoundError
    ):

        FBRVectorStore.load(
            tmp_path
        )
