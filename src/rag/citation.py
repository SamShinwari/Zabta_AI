def format_citation(result: dict) -> str:
    """
    Convert one retrieved FBR result into
    a human-readable citation.
    """

    metadata = result.get(
        "metadata",
        {},
    )

    source = metadata.get(
        "source",
        "Unknown FBR document",
    )

    page = metadata.get(
        "page"
    )

    chunk = metadata.get(
        "chunk"
    )

    parts = [source]

    if page is not None:
        parts.append(
            f"Page {page}"
        )

    if chunk is not None:
        parts.append(
            f"Chunk {chunk}"
        )

    return " — ".join(parts)


def add_citation(
    result: dict,
) -> dict:
    """
    Add formatted citation information
    to a retrieved result.
    """

    result = result.copy()

    result["citation"] = format_citation(
        result
    )

    return result
