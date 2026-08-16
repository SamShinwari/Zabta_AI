from pathlib import Path

from src.fbr.pdf_processor import (
    FBRPDFProcessor,
)


FBR_ROOT = Path(
    "data/raw/fbr"
)


def test_find_pdfs():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    pdfs = processor.find_pdfs()

    assert len(pdfs) == 187

    assert all(
        pdf.suffix.lower() == ".pdf"
        for pdf in pdfs
    )


def test_category():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    path = (
        FBR_ROOT
        / "acts"
        / "test.pdf"
    )

    assert (
        processor.get_category(path)
        == "acts"
    )


def test_category_rules():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    path = (
        FBR_ROOT
        / "rules"
        / "test.pdf"
    )

    assert (
        processor.get_category(path)
        == "rules"
    )


def test_clean_text():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    text = (
        "Hello     world.\n\n\n"
        "This   is   a test."
    )

    cleaned = processor.clean_text(
        text
    )

    assert cleaned == (
        "Hello world.\n\n"
        "This is a test."
    )


def test_empty_text():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    assert (
        processor.clean_text("")
        == ""
    )

    assert (
        processor.clean_text("   ")
        == ""
    )


def test_process_real_pdf():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    pdfs = processor.find_pdfs()

    processed = None

    for pdf in pdfs:

        pages = processor.process_pdf(
            pdf
        )

        if pages:

            processed = pages
            break

    assert processed is not None

    page = processed[0]

    assert page.source.endswith(
        ".pdf"
    )

    assert page.page_number >= 1

    assert page.total_pages >= (
        page.page_number
    )

    assert page.text.strip()

    assert page.category != "unknown"


def test_page_metadata():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    pdfs = processor.find_pdfs()

    for pdf in pdfs:

        pages = processor.process_pdf(
            pdf
        )

        if not pages:
            continue

        page = pages[0]

        assert page.source == pdf.name

        assert page.page_number >= 1

        assert (
            page.total_pages
            >= page.page_number
        )

        break

    else:

        raise AssertionError(
            "No PDF pages were processed"
        )


def test_to_dict():

    processor = FBRPDFProcessor(
        FBR_ROOT
    )

    pdfs = processor.find_pdfs()

    for pdf in pdfs:

        pages = processor.process_pdf(
            pdf
        )

        if not pages:
            continue

        data = processor.to_dict(
            pages[0]
        )

        assert isinstance(
            data,
            dict
        )

        assert "source" in data
        assert "page_number" in data
        assert "text" in data
        assert "category" in data
        assert "total_pages" in data

        break

    else:

        raise AssertionError(
            "No processed PDF found"
        )
