from __future__ import annotations

from typing import Any


class FBRCitationBuilder:
    """
    Builds application-controlled citations for FBR sources.
    """

    @staticmethod
    def build(
        result: dict[str, Any],
        citation_number: int | None = None,
    ) -> dict[str, Any]:

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get(
            "page",
        )

        chunk = metadata.get(
            "chunk",
        )

        if page is not None:
            citation = (
        
                f"{source}, p. {page}"
            )
        else:
            citation = source

        return {
            "id": citation_number,
            "citation": citation,
            "source": source,
            "page": page,
            "chunk": chunk,
            "source_path": metadata.get(
                "source_path",
            ),
            "relative_path": metadata.get(
                "relative_path",
            ),
            "score": result.get(
                "score",
            ),
            "text": result.get(
                "text",
                "",
            ),
        }

    @staticmethod
    def build_many(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        citations = []

        seen: set[tuple[Any, Any]] = set()

        for result in results:

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
            )

            page = metadata.get(
                "page",
            )

            key = (
                source,
                page,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                FBRCitationBuilder.build(
                    result,
                    citation_number=len(citations) + 1,
                )
            )

        return citations
    