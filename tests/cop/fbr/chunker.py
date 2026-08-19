from dataclasses import dataclass, asdict
from pathlib import Path
import re


# ============================================================
# Document Chunk
# ============================================================

@dataclass
class FBRChunk:
    """
    A retrieval-ready chunk extracted from an FBR document.
    """

    chunk_id: str
    text: str

    source: str
    category: str

    page_start: int
    page_end: int

    chunk_index: int

    char_count: int

    metadata: dict

    def to_dict(self) -> dict:
        """
        Convert chunk to dictionary.
        """

        return asdict(self)


# ============================================================
# FBR Chunker
# ============================================================

class FBRChunker:
    """
    Split processed FBR document text into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # --------------------------------------------------------
    # Text Cleaning
    # --------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalize whitespace while preserving readable text.
        """

        if not text:
            return ""

        # Normalize line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Collapse excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # --------------------------------------------------------
    # Page-Aware Chunking
    # --------------------------------------------------------

    def chunk_pages(
        self,
        pages: list[dict],
        source: str,
        category: str,
    ) -> list[FBRChunk]:
        """
        Create chunks from page-level extracted content.

        Expected page format:

        {
            "page_number": 1,
            "text": "..."
        }
        """

        chunks = []

        current_text = ""
        current_start_page = None
        current_end_page = None

        chunk_index = 0

        for page in pages:

            page_number = int(
                page["page_number"]
            )

            page_text = self.clean_text(
                page.get("text", "")
            )

            if not page_text:
                continue

            if current_start_page is None:
                current_start_page = page_number

            current_end_page = page_number

            # ------------------------------------------------
            # Add page text to current buffer.
            # ------------------------------------------------

            if current_text:

                current_text += "\n\n"

            current_text += page_text

            # ------------------------------------------------
            # Create chunks when buffer is large enough.
            # ------------------------------------------------

            while len(current_text) >= self.chunk_size:

                chunk_text = current_text[
                    :self.chunk_size
                ].strip()

                if chunk_text:

                    chunk_id = self._make_chunk_id(
                        source=source,
                        chunk_index=chunk_index,
                    )

                    chunks.append(
                        FBRChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            source=source,
                            category=category,
                            page_start=current_start_page,
                            page_end=current_end_page,
                            chunk_index=chunk_index,
                            char_count=len(chunk_text),
                            metadata={
                                "source": source,
                                "category": category,
                                "page_start": current_start_page,
                                "page_end": current_end_page,
                            },
                        )
                    )

                    chunk_index += 1

                # ------------------------------------------------
                # Keep overlap for retrieval continuity.
                # ------------------------------------------------

                overlap_start = max(
                    0,
                    self.chunk_size - self.chunk_overlap,
                )

                current_text = current_text[
                    overlap_start:
                ].strip()

                # The remaining text belongs to the
                # current/end page context.
                current_start_page = current_end_page

        # ----------------------------------------------------
        # Final remaining chunk.
        # ----------------------------------------------------

        if current_text.strip():

            chunk_text = current_text.strip()

            chunk_id = self._make_chunk_id(
                source=source,
                chunk_index=chunk_index,
            )

            chunks.append(
                FBRChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    category=category,
                    page_start=current_start_page,
                    page_end=current_end_page,
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    metadata={
                        "source": source,
                        "category": category,
                        "page_start": current_start_page,
                        "page_end": current_end_page,
                    },
                )
            )

        return chunks

    # --------------------------------------------------------
    # Simple Text Chunking
    # --------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        source: str = "",
        category: str = "unknown",
    ) -> list[FBRChunk]:
        """
        Chunk a plain text document.
        """

        text = self.clean_text(text)

        if not text:
            return []

        pages = [
            {
                "page_number": 1,
                "text": text,
            }
        ]

        return self.chunk_pages(
            pages=pages,
            source=source,
            category=category,
        )

    # --------------------------------------------------------
    # Chunk ID
    # --------------------------------------------------------

    @staticmethod
    def _make_chunk_id(
        source: str,
        chunk_index: int,
    ) -> str:
        """
        Create a stable chunk identifier.
        """

        source_name = Path(
            source
        ).stem

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            source_name,
        )

        return (
            f"{safe_name}"
            f"_chunk_{chunk_index:05d}"
        )
