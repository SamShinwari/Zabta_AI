from __future__ import annotations

from typing import Any


def build_context(
    results: list[dict[str, Any]],
    max_chars: int = 12000,
) -> str:
    """
    Convert retrieved FBR results into structured context
    for the answer-generation layer.

    Each result contains:
        - text
        - score
        - metadata
        - citation
    """

    if not results:
        return "No relevant FBR sources were retrieved."

    context_parts: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, start=1):

        text = result.get("text", "").strip()

        if not text:
            continue

        metadata = result.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown FBR document",
        )

        page = metadata.get("page")
        chunk = metadata.get("chunk")
        score = result.get("score")

        citation = result.get(
            "citation",
            source,
        )

        section = (
            f"SOURCE {i}\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk}\n"
            f"Similarity Score: {score:.4f}\n"
            f"Citation: {citation}\n"
            f"Text:\n{text}\n"
        )

        if total_chars + len(section) > max_chars:
            remaining = max_chars - total_chars

            if remaining > 200:
                section = section[:remaining]

                context_parts.append(section)

            break

        context_parts.append(section)
        total_chars += len(section)

    return "\n" + "\n".join(context_parts)
