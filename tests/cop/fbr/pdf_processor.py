from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import re

from src.fbr.pdf_extractor import (
    FBRPDFExtractor,
    PDFExtractionError,
)


# ============================================================
# Processed Page
# ============================================================

@dataclass
class ProcessedPage:
    """
    Cleaned text extracted from one FBR PDF page.
    """

    source: str
    page_number: int
    text: str
    category: str
    total_pages: int


# ============================================================
# PDF Processor
# ============================================================

class FBRPDFProcessor:
    """
    Process FBR PDFs into clean page-level text records.
    """

    def __init__(
        self,
        root_directory: str | Path
    ):
        self.root_directory = Path(
            root_directory
        )

        self.extractor = FBRPDFExtractor()

    # ========================================================
    # Find PDFs
    # ========================================================

    def find_pdfs(self) -> list[Path]:
        """
        Find all PDF files recursively.
        """

        if not self.root_directory.exists():
            return []

        return sorted(
            self.root_directory.rglob("*.pdf")
        )

    # ========================================================
    # Category
    # ========================================================

    def get_category(
        self,
        pdf_path: str | Path
    ) -> str:
        """
        Determine the top-level FBR document category.

        Example:

            data/raw/fbr/acts/file.pdf

        returns:

            acts
        """

        pdf_path = Path(pdf_path)

        try:

            relative_path = pdf_path.relative_to(
                self.root_directory
            )

        except ValueError:

            return "unknown"

        parts = relative_path.parts

        if len(parts) >= 2:
            return parts[0]

        return "unknown"

    # ========================================================
    # Clean Text
    # ========================================================

    @staticmethod
    def clean_text(
        text: str
    ) -> str:
        """
        Clean extracted PDF text.

        Keeps meaningful line breaks while removing
        excessive whitespace.
        """

        if not text:
            return ""

        # Normalize line endings.
        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        # Remove trailing spaces.
        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        # Replace tabs and repeated spaces.
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Collapse 3+ consecutive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ========================================================
    # Process One PDF
    # ========================================================

    def process_pdf(
        self,
        pdf_path: str | Path
    ) -> list[ProcessedPage]:
        """
        Extract and clean one PDF.

        Pages without text are skipped.

        Failed PDFs return an empty list.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return []

        if not pdf_path.is_file():
            return []

        if pdf_path.suffix.lower() != ".pdf":
            return []

        try:

            extracted = self.extractor.extract(
                pdf_path
            )

        except PDFExtractionError:

            return []

        except Exception:

            return []

        category = self.get_category(
            pdf_path
        )

        processed_pages = []

        for page in extracted.pages:

            cleaned_text = self.clean_text(
                page.text
            )

            # Skip pages with no usable text.
            if not cleaned_text:
                continue

            processed_pages.append(
                ProcessedPage(
                    source=pdf_path.name,
                    page_number=page.page_number,
                    text=cleaned_text,
                    category=category,
                    total_pages=extracted.page_count,
                )
            )

        return processed_pages

    # ========================================================
    # Process Directory
    # ========================================================

    def process_directory(
        self
    ) -> list[ProcessedPage]:
        """
        Process all PDFs under the root directory.
        """

        all_pages = []

        for pdf_path in self.find_pdfs():

            pages = self.process_pdf(
                pdf_path
            )

            all_pages.extend(
                pages
            )

        return all_pages

    # ========================================================
    # Convert to Dictionary
    # ========================================================

    @staticmethod
    def to_dict(
        page: ProcessedPage
    ) -> dict:
        """
        Convert ProcessedPage into a dictionary.
        """

        return asdict(page)