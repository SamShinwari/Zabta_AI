from dataclasses import dataclass
from pathlib import Path

import fitz


# ============================================================
# Extracted Page
# ============================================================

@dataclass
class ExtractedPage:
    """
    Text extracted from one PDF page.
    """

    page_number: int
    text: str


# ============================================================
# Extracted PDF
# ============================================================

@dataclass
class ExtractedPDF:
    """
    Complete extracted content of a PDF.
    """

    path: Path
    page_count: int
    pages: list[ExtractedPage]

    @property
    def text(self) -> str:
        """
        Return complete PDF text.
        """

        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )

    @property
    def text_page_count(self) -> int:
        """
        Number of pages containing extractable text.
        """

        return sum(
            1
            for page in self.pages
            if page.text.strip()
        )


# ============================================================
# PDF Extraction Error
# ============================================================

class PDFExtractionError(Exception):
    """
    Raised when a PDF cannot be opened or processed.
    """

    pass


# ============================================================
# PDF Extractor
# ============================================================

class FBRPDFExtractor:
    """
    Extract text from FBR PDF documents using PyMuPDF.
    """

    def extract(
        self,
        pdf_path: str | Path
    ) -> ExtractedPDF:

        pdf_path = Path(pdf_path)

        # ----------------------------------------------------
        # Validate file
        # ----------------------------------------------------

        if not pdf_path.exists():

            raise PDFExtractionError(
                f"PDF does not exist: {pdf_path}"
            )

        if not pdf_path.is_file():

            raise PDFExtractionError(
                f"Path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise PDFExtractionError(
                f"Not a PDF file: {pdf_path}"
            )

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        try:

            document = fitz.open(
                pdf_path
            )

        except Exception as exc:

            raise PDFExtractionError(
                f"Could not open PDF "
                f"{pdf_path}: {exc}"
            ) from exc

        pages = []

        try:

            page_count = len(document)

            # ------------------------------------------------
            # Extract page-by-page
            # ------------------------------------------------

            for page_index in range(
                page_count
            ):

                page = document[
                    page_index
                ]

                try:

                    text = page.get_text(
                        "text"
                    )

                except Exception as exc:

                    text = ""

                    print(
                        f"Warning: could not extract "
                        f"page {page_index + 1} "
                        f"from {pdf_path}: {exc}"
                    )

                # Normalize line endings.
                text = text.replace(
                    "\r\n",
                    "\n"
                )

                text = text.replace(
                    "\r",
                    "\n"
                )

                pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

            return ExtractedPDF(
                path=pdf_path,
                page_count=page_count,
                pages=pages,
            )

        finally:

            document.close()
