"""
PDF quality analysis for FBR documents.

Classifies PDFs into:
    - TEXT: PDF opens and contains extractable text
    - SCANNED: PDF opens but contains no extractable text
    - PARTIAL_TEXT: PDF opens and only some pages contain text
    - FAILED: PDF cannot be opened/read

The module is intentionally independent from the RAG pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class PDFQualityResult:
    """Quality information for a single PDF."""

    file_path: str
    file_name: str
    pages: int
    text_pages: int
    empty_pages: int
    status: str
    error: Optional[str] = None

    @property
    def text_ratio(self) -> float:
        """Return the proportion of pages containing text."""
        if self.pages == 0:
            return 0.0

        return round(self.text_pages / self.pages, 4)


class PDFQualityAnalyzer:
    """
    Analyze the quality of FBR PDF files.

    Parameters
    ----------
    root_dir:
        Directory containing FBR PDFs.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def analyze_pdf(self, pdf_path: str | Path) -> PDFQualityResult:
        """
        Analyze a single PDF.

        Classification:
            TEXT          -> all pages contain text
            SCANNED       -> no pages contain text
            PARTIAL_TEXT  -> some pages contain text
            FAILED        -> PDF cannot be opened/read
        """

        pdf_path = Path(pdf_path)

        try:
            doc = fitz.open(pdf_path)

            total_pages = len(doc)
            text_pages = 0

            for page in doc:
                text = page.get_text("text").strip()

                if text:
                    text_pages += 1

            doc.close()

            empty_pages = total_pages - text_pages

            if total_pages == 0:
                status = "FAILED"
                error = "PDF contains zero pages"

            elif text_pages == 0:
                status = "SCANNED"
                error = None

            elif text_pages == total_pages:
                status = "TEXT"
                error = None

            else:
                status = "PARTIAL_TEXT"
                error = None

            return PDFQualityResult(
                file_path=str(pdf_path),
                file_name=pdf_path.name,
                pages=total_pages,
                text_pages=text_pages,
                empty_pages=empty_pages,
                status=status,
                error=error,
            )

        except Exception as exc:
            return PDFQualityResult(
                file_path=str(pdf_path),
                file_name=pdf_path.name,
                pages=0,
                text_pages=0,
                empty_pages=0,
                status="FAILED",
                error=str(exc),
            )

    def find_pdfs(self) -> List[Path]:
        """Find all PDF files recursively under root_dir."""

        if not self.root_dir.exists():
            return []

        return sorted(
            self.root_dir.rglob("*.pdf"),
            key=lambda path: str(path).lower(),
        )

    def analyze_all(self) -> List[PDFQualityResult]:
        """Analyze every PDF under root_dir."""

        results = []

        for pdf_path in self.find_pdfs():
            results.append(self.analyze_pdf(pdf_path))

        return results

    @staticmethod
    def summarize(
        results: List[PDFQualityResult],
    ) -> dict:
        """Create aggregate statistics from PDF quality results."""

        total_pdfs = len(results)

        total_pages = sum(result.pages for result in results)
        total_text_pages = sum(result.text_pages for result in results)
        total_empty_pages = sum(result.empty_pages for result in results)

        text_pdfs = sum(
            result.status == "TEXT"
            for result in results
        )

        scanned_pdfs = sum(
            result.status == "SCANNED"
            for result in results
        )

        partial_pdfs = sum(
            result.status == "PARTIAL_TEXT"
            for result in results
        )

        failed_pdfs = sum(
            result.status == "FAILED"
            for result in results
        )

        return {
            "total_pdfs": total_pdfs,
            "text_pdfs": text_pdfs,
            "scanned_pdfs": scanned_pdfs,
            "partial_text_pdfs": partial_pdfs,
            "failed_pdfs": failed_pdfs,
            "total_pages": total_pages,
            "text_pages": total_text_pages,
            "empty_pages": total_empty_pages,
        }

    @staticmethod
    def save_report(
        results: List[PDFQualityResult],
        output_path: str | Path,
    ) -> None:
        """Save detailed quality report as JSON."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = PDFQualityAnalyzer.summarize(results)

        report = {
            "summary": summary,
            "documents": [
                asdict(result)
                | {"text_ratio": result.text_ratio}
                for result in results
            ],
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )
