from pathlib import Path

from src.fbr.pdf_quality import PDFQualityAnalyzer


FBR_DIR = Path("data/raw/fbr")


def test_find_pdfs():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    pdfs = analyzer.find_pdfs()

    assert len(pdfs) == 187


def test_analyze_real_pdf():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    pdf = (
        FBR_DIR
        / "acts"
        / "FBR_Sales_Tax_Act_PDFs"
        / "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    result = analyzer.analyze_pdf(pdf)

    assert result.status == "TEXT"
    assert result.pages == 229
    assert result.text_pages == 229
    assert result.empty_pages == 0


def test_analyze_scanned_pdf():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    pdf = (
        FBR_DIR
        / "acts"
        / "Benami Transactions (Prohibition) Act, 2017.pdf"
    )

    result = analyzer.analyze_pdf(pdf)

    assert result.status == "SCANNED"
    assert result.pages == 26
    assert result.text_pages == 0


def test_analyze_failed_pdf():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    pdf = (
        FBR_DIR
        / "ordinances"
        / "Income Tax Ordinance, 1979 - Old Laws.pdf"
    )

    result = analyzer.analyze_pdf(pdf)

    assert result.status == "FAILED"


def test_text_ratio():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    pdf = (
        FBR_DIR
        / "acts"
        / "FBR_Sales_Tax_Act_PDFs"
        / "Sales Tax Act 1990 amended upto 30-06-2026.pdf"
    )

    result = analyzer.analyze_pdf(pdf)

    assert result.text_ratio == 1.0


def test_summary():
    analyzer = PDFQualityAnalyzer(FBR_DIR)

    results = analyzer.analyze_all()
    summary = analyzer.summarize(results)

    assert summary["total_pdfs"] == 187
    assert summary["total_pages"] == 31004
