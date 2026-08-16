from pathlib import Path

from src.fbr.pdf_scanner import (
    FBRPDFScanner,
)


def create_test_files(
    directory: Path
):

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # PDF files
    (directory / "FinanceAct2026.pdf").write_bytes(
        b"%PDF-test"
    )

    (directory / "SalesTaxAct.pdf").write_bytes(
        b"%PDF-test"
    )

    # Nested PDF
    nested = directory / "sros"
    nested.mkdir()

    (nested / "SRO1234.pdf").write_bytes(
        b"%PDF-test"
    )

    # Non-PDF file
    (directory / "notes.txt").write_text(
        "This is not a PDF."
    )


def test_scan_pdf_files(tmp_path):

    fbr_directory = (
        tmp_path / "fbr_docs"
    )

    create_test_files(
        fbr_directory
    )

    scanner = FBRPDFScanner(
        fbr_directory
    )

    results = scanner.scan()

    assert len(results) == 3


def test_scan_filenames(tmp_path):

    fbr_directory = (
        tmp_path / "fbr_docs"
    )

    create_test_files(
        fbr_directory
    )

    scanner = FBRPDFScanner(
        fbr_directory
    )

    results = scanner.scan()

    filenames = {
        item.filename
        for item in results
    }

    assert "FinanceAct2026.pdf" in filenames

    assert "SalesTaxAct.pdf" in filenames

    assert "SRO1234.pdf" in filenames


def test_pdf_count(tmp_path):

    fbr_directory = (
        tmp_path / "fbr_docs"
    )

    create_test_files(
        fbr_directory
    )

    scanner = FBRPDFScanner(
        fbr_directory
    )

    assert scanner.count() == 3


def test_missing_directory(tmp_path):

    missing_directory = (
        tmp_path / "does_not_exist"
    )

    scanner = FBRPDFScanner(
        missing_directory
    )

    try:

        scanner.scan()

        assert False, (
            "Expected FileNotFoundError"
        )

    except FileNotFoundError:

        assert True
