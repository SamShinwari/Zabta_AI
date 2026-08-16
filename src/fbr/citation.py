from __future__ import annotations

from typing import Any


class FBRCitationBuilder:
    """
    Builds clean citations from FBR retrieval results.

    Expected retrieval result format:

    {
        "rank": 1,
        "score": 0.68,
        "text": "...",
        "metadata": {
            "source": "Sales Tax Act 1990 amended upto 30-06-2026.pdf",
            "source_path": "...",
            "relative_path": "...",
            "page": 28,
            "chunk": 1
        }
    }
    """

    def build(
        self,
        result: dict[str, Any],
        citation_number: int,
    ) -> dict[str, Any]:
        metadata = result.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get("page")

        chunk = metadata.get("chunk")

        if page is not None:
            citation_text = (
                f"[{citation_number}] "
                f"{source}, p. {page}"
            )
        else:
            citation_text = (
                f"[{citation_number}] "
                f"{source}"
            )

        return {
            "id": citation_number,
            "citation": citation_text,
            "source": source,
            "page": page,
            "chunk": chunk,
            "source_path": metadata.get(
                "source_path"
            ),
            "relative_path": metadata.get(
                "relative_path"
            ),
            "score": result.get("score"),
            "text": result.get("text", ""),
        }

    def build_many(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build citations while removing duplicate
        source/page combinations.
        """

        citations = []

        seen: set[tuple[Any, Any]] = set()

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source"
            )

            page = metadata.get(
                "page"
            )

            key = (
                source,
                page,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                self.build(
                    result,
                    len(citations) + 1,
                )
            )

        return citations
