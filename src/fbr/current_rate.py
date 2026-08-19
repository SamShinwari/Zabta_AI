from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CurrentRateResult:
    """
    Represents the sales tax rate resolved from FBR evidence.

    The rate is NOT hard-coded here.
    It is produced by the FBR retrieval/rate-resolution layer.
    """

    rate: float
    category: str
    source_document: str
    page: int | None
    chunk: int | None
    confidence: float
    text: str

    @property
    def source_reference(self) -> str:
        """
        Human-readable FBR source reference.
        """

        parts = [
            self.source_document
        ]

        if self.page is not None:
            parts.append(
                f"Page {self.page}"
            )

        if self.chunk is not None:
            parts.append(
                f"Chunk {self.chunk}"
            )

        return " — ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result into a serializable dictionary.
        """

        return {
            "rate": self.rate,
            "category": self.category,
            "source_document": self.source_document,
            "page": self.page,
            "chunk": self.chunk,
            "confidence": self.confidence,
            "text": self.text,
            "source_reference": self.source_reference,
        }