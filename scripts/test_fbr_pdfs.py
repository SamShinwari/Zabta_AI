import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
from src.fbr.pdf_scanner import FBRPDFScanner
from src.fbr.pdf_extractor import (
    FBRPDFExtractor,
    PDFExtractionError,
)


FBR_DIRECTORY = Path(
    "data/raw/fbr"
)


def main():

    scanner = FBRPDFScanner(
        FBR_DIRECTORY
    )

    extractor = FBRPDFExtractor()

    pdfs = scanner.scan()

    print("=" * 60)

    print(
        f"PDF files found: {len(pdfs)}"
    )

    print("=" * 60)

    successful = 0
    failed = 0

    total_pages = 0
    text_pages = 0

    for pdf in pdfs:

        print(
            f"\nProcessing: {pdf.path}"
        )

        try:

            result = extractor.extract(
                pdf.path
            )

            successful += 1

            total_pages += (
                result.page_count
            )

            text_pages += (
                result.text_page_count
            )

            print(
                f"  Pages: {result.page_count}"
            )

            print(
                f"  Text pages: "
                f"{result.text_page_count}"
            )

        except PDFExtractionError as exc:

            failed += 1

            print(
                f"  ERROR: {exc}"
            )

    print("\n" + "=" * 60)

    print(
        f"Total PDFs: {len(pdfs)}"
    )

    print(
        f"Successfully loaded: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total pages: {total_pages}"
    )

    print(
        f"Pages with text: {text_pages}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()
