from pathlib import Path

from src.fbr.pdf_quality import PDFQualityAnalyzer


def main():
    root_dir = Path("data/raw/fbr")

    output_file = Path(
        "data/raw/fbr/pdf_quality_report.json"
    )

    analyzer = PDFQualityAnalyzer(root_dir)

    print("=" * 60)
    print("FBR PDF QUALITY ANALYSIS")
    print("=" * 60)

    results = analyzer.analyze_all()

    summary = analyzer.summarize(results)

    print(f"Total PDFs       : {summary['total_pdfs']}")
    print(f"Text PDFs        : {summary['text_pdfs']}")
    print(f"Scanned PDFs     : {summary['scanned_pdfs']}")
    print(f"Partial PDFs     : {summary['partial_text_pdfs']}")
    print(f"Failed PDFs      : {summary['failed_pdfs']}")
    print(f"Total pages      : {summary['total_pages']}")
    print(f"Text pages       : {summary['text_pages']}")
    print(f"Empty pages      : {summary['empty_pages']}")

    analyzer.save_report(
        results,
        output_file,
    )

    print("=" * 60)
    print(f"Report saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
