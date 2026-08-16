import fitz
import pytest

from src.fbr.pdf_extractor import (
    FBRPDFExtractor,
    PDFExtractionError,
)


# ============================================================
# Create Test PDF
# ============================================================

def create_test_pdf(path):

    document = fitz.open()

    page1 = document.new_page()

    page1.insert_text(
        (72, 72),
        "Sales Tax Act 1990"
    )

    page1.insert_text(
        (72, 100),
        "Sales tax shall be charged."
    )

    page2 = document.new_page()

    page2.insert_text(
        (72, 72),
        "Finance Act 2026"
    )

    page2.insert_text(
        (72, 100),
        "Effective from 1 July 2026."
    )

    document.save(path)

    document.close()


# ============================================================
# Test Basic Extraction
# ============================================================

def test_pdf_extraction(tmp_path):

    pdf_path = (
        tmp_path / "test.pdf"
    )

    create_test_pdf(
        pdf_path
    )

    extractor = FBRPDFExtractor()

    result = extractor.extract(
        pdf_path
    )

    assert result.page_count == 2

    assert result.text_page_count == 2

    assert len(result.pages) == 2


# ============================================================
# Test Page Numbers
# ============================================================

def test_page_numbers(tmp_path):

    pdf_path = (
        tmp_path / "test.pdf"
    )

    create_test_pdf(
        pdf_path
    )

    extractor = FBRPDFExtractor()

    result = extractor.extract(
        pdf_path
    )

    assert result.pages[0].page_number == 1

    assert result.pages[1].page_number == 2


# ============================================================
# Test Extracted Text
# ============================================================

def test_extracted_text(tmp_path):

    pdf_path = (
        tmp_path / "test.pdf"
    )

    create_test_pdf(
        pdf_path
    )

    extractor = FBRPDFExtractor()

    result = extractor.extract(
        pdf_path
    )

    assert (
        "Sales Tax Act 1990"
        in result.pages[0].text
    )

    assert (
        "Finance Act 2026"
        in result.pages[1].text
    )


# ============================================================
# Test Complete Text
# ============================================================

def test_complete_text(tmp_path):

    pdf_path = (
        tmp_path / "test.pdf"
    )

    create_test_pdf(
        pdf_path
    )

    extractor = FBRPDFExtractor()

    result = extractor.extract(
        pdf_path
    )

    assert "Sales Tax Act 1990" in result.text

    assert "Finance Act 2026" in result.text

    assert (
        "Effective from 1 July 2026."
        in result.text
    )


# ============================================================
# Test Missing PDF
# ============================================================

def test_missing_pdf(tmp_path):

    pdf_path = (
        tmp_path / "missing.pdf"
    )

    extractor = FBRPDFExtractor()

    with pytest.raises(
        PDFExtractionError
    ):

        extractor.extract(
            pdf_path
        )


# ============================================================
# Test Non-PDF File
# ============================================================

def test_non_pdf_file(tmp_path):

    text_file = (
        tmp_path / "test.txt"
    )

    text_file.write_text(
        "This is not a PDF."
    )

    extractor = FBRPDFExtractor()

    with pytest.raises(
        PDFExtractionError
    ):

        extractor.extract(
            text_file
        )
