from src.fbr.chunker import FBRChunker


def test_clean_text():

    chunker = FBRChunker()

    text = "Hello    world\n\n\nTax   Rules"

    cleaned = chunker.clean_text(text)

    assert cleaned == "Hello world\n\nTax Rules"


def test_empty_text():

    chunker = FBRChunker()

    chunks = chunker.chunk_text("")

    assert chunks == []


def test_chunk_text():

    chunker = FBRChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "A" * 250

    chunks = chunker.chunk_text(
        text,
        source="Sales Tax Act 1990.pdf",
        category="acts",
    )

    assert len(chunks) > 1


def test_chunk_metadata():

    chunker = FBRChunker(
        chunk_size=50,
        chunk_overlap=10,
    )

    pages = [
        {
            "page_number": 1,
            "text": "Sales Tax Act " * 10,
        },
        {
            "page_number": 2,
            "text": "Input tax adjustment " * 10,
        },
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        source="Sales Tax Act 1990.pdf",
        category="acts",
    )

    assert len(chunks) > 0

    chunk = chunks[0]

    assert chunk.source == "Sales Tax Act 1990.pdf"
    assert chunk.category == "acts"
    assert chunk.page_start >= 1
    assert chunk.page_end >= chunk.page_start


def test_chunk_id():

    chunker = FBRChunker()

    chunks = chunker.chunk_text(
        "This is some FBR content.",
        source="Sales Tax Act 1990.pdf",
        category="acts",
    )

    assert chunks[0].chunk_id.startswith(
        "Sales_Tax_Act_1990_chunk_"
    )


def test_to_dict():

    chunker = FBRChunker()

    chunks = chunker.chunk_text(
        "FBR tax content.",
        source="test.pdf",
        category="acts",
    )

    data = chunks[0].to_dict()

    assert isinstance(data, dict)
    assert "chunk_id" in data
    assert "text" in data
    assert "metadata" in data
