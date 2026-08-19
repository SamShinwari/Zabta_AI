from dataclasses import dataclass
from pathlib import Path


# ============================================================
# Supported Document Extensions
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
}


# ============================================================
# PDF Information
# ============================================================

@dataclass
class PDFFile:
    """
    Information about a discovered PDF file.
    """

    path: Path
    filename: str
    size_bytes: int


# ============================================================
# PDF Scanner
# ============================================================

class FBRPDFScanner:
    """
    Scan the FBR document directory and discover PDF files.
    """

    def __init__(
        self,
        root_directory: str | Path
    ):

        self.root_directory = Path(
            root_directory
        )


    # ========================================================
    # Scan PDFs
    # ========================================================

    def scan(self) -> list[PDFFile]:
        """
        Recursively scan the FBR directory for PDF files.
        """

        if not self.root_directory.exists():

            raise FileNotFoundError(
                "FBR document directory does not exist: "
                f"{self.root_directory}"
            )

        results = []

        for path in self.root_directory.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            results.append(
                PDFFile(
                    path=path,
                    filename=path.name,
                    size_bytes=path.stat().st_size,
                )
            )

        # Sort for deterministic results.
        results.sort(
            key=lambda item: str(item.path).lower()
        )

        return results


    # ========================================================
    # Count PDFs
    # ========================================================

    def count(self) -> int:
        """
        Return number of discovered PDF files.
        """

        return len(
            self.scan()
        )
