from src.fbr.citation import FBRCitationBuilder


def make_result(
    source="Sales Tax Act 1990 amended upto 30-06-2026.pdf",
    page=28,
    chunk=1,
):
    return {
        "rank": 1,
        "score": 0.68,
        "text": "Sales tax shall be charged at eighteen per cent.",
        "metadata": {
            "source": source,
            "source_path": "/some/path/document.pdf",
            "relative_path": "data/raw/fbr/acts/document.pdf",
            "page": page,
            "chunk": chunk,
        },
    }


def test_build_citation():
    builder = FBRCitationBuilder()

    citation = builder.build(
        make_result(),
        1,
    )

    assert citation["id"] == 1

    assert (
        citation["citation"]
        == "[1] Sales Tax Act 1990 amended upto 30-06-2026.pdf, p. 28"
    )

    assert citation["source"] == (
        "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    assert citation["page"] == 28


def test_build_citation_without_page():
    builder = FBRCitationBuilder()

    result = make_result(page=None)

    citation = builder.build(
        result,
        1,
    )

    assert citation["citation"] == (
        "[1] Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )


def test_build_many():
    builder = FBRCitationBuilder()

    results = [
        make_result(page=28),
        make_result(page=29),
    ]

    citations = builder.build_many(
        results
    )

    assert len(citations) == 2

    assert citations[0]["id"] == 1
    assert citations[1]["id"] == 2


def test_duplicate_citations_removed():
    builder = FBRCitationBuilder()

    results = [
        make_result(page=28, chunk=1),
        make_result(page=28, chunk=2),
        make_result(page=28, chunk=3),
    ]

    citations = builder.build_many(
        results
    )

    assert len(citations) == 1


def test_different_pages_are_kept():
    builder = FBRCitationBuilder()

    results = [
        make_result(page=28),
        make_result(page=29),
    ]

    citations = builder.build_many(
        results
    )

    assert len(citations) == 2


def test_citation_preserves_text():
    builder = FBRCitationBuilder()

    result = make_result()

    citation = builder.build(
        result,
        1,
    )

    assert citation["text"] == (
        "Sales tax shall be charged at eighteen per cent."
    )
