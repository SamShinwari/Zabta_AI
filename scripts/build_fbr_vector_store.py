#!/usr/bin/env python3

"""
Zabta - FBR Vector Store Builder

Pipeline:

    FBR PDFs
        ↓
    PDF discovery
        ↓
    PDF text extraction
        ↓
    Text chunking
        ↓
    BGE embeddings
        ↓
    FAISS vector index
        ↓
    Metadata + pipeline result

Input:
    data/raw/fbr/**/*.pdf

Output:
    data/vector_database/fbr/
        ├── index.faiss
        ├── metadata.json
        └── pipeline_result.json

Embedding model:
    BAAI/bge-m3
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import faiss
import pymupdf

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "data" / "raw" / "fbr"

VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_database"
    / "fbr"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

BATCH_SIZE = 32

NORMALIZE_EMBEDDINGS = True

MIN_TEXT_LENGTH = 20


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger("zabta_fbr_vector_store")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header() -> None:
    """Print pipeline header."""

    print("=" * 70)
    print("ZABTA - FBR VECTOR STORE BUILD")
    print("=" * 70)

    print(f"PDF directory:    {PDF_DIR}")
    print(f"Vector directory: {VECTOR_DIR}")
    print(f"Embedding model:  {EMBEDDING_MODEL}")
    print(f"Chunk size:       {CHUNK_SIZE}")
    print(f"Chunk overlap:    {CHUNK_OVERLAP}")
    print("=" * 70)


def discover_pdfs() -> list[Path]:
    """
    Recursively discover all FBR PDFs.
    """

    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"FBR PDF directory does not exist:\n{PDF_DIR}"
        )

    pdfs = sorted(
        [
            p
            for p in PDF_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() == ".pdf"
        ],
        key=lambda p: str(p).lower(),
    )

    return pdfs


def clean_text(text: str) -> str:
    """
    Basic text cleanup.

    Does not aggressively modify legal text because
    preserving the original wording is important for
    FBR documents.
    """

    if not text:
        return ""

    # Normalize whitespace while preserving paragraph content.
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def extract_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """
    Extract text from a PDF page by page.

    Returns:

        pages:
            List containing page-level text and metadata.

        error:
            None if successful, otherwise an error message.
    """

    pages: list[dict[str, Any]] = []

    document = None

    try:

        document = pymupdf.open(str(pdf_path))

        page_count = len(document)

        for page_number in range(page_count):

            try:

                page = document.load_page(page_number)

                text = page.get_text("text")

                text = clean_text(text)

                if text:

                    pages.append(
                        {
                            "page": page_number + 1,
                            "text": text,
                        }
                    )

            except Exception as page_error:

                logger.warning(
                    "Page extraction failed: %s | page=%s | %s",
                    pdf_path.name,
                    page_number + 1,
                    page_error,
                )

        return pages, None

    except Exception as error:

        return [], str(error)

    finally:

        if document is not None:

            try:
                document.close()
            except Exception:
                pass


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Create character-based overlapping chunks.

    Example:

        chunk size = 500
        overlap    = 100

    The next chunk starts 400 characters after
    the previous chunk.
    """

    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    step = chunk_size - overlap

    chunks: list[str] = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += step

    return chunks


def build_document_chunks(
    pdf_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Extract a PDF and convert it into chunks.

    Returns:

        chunks:
            Chunk records.

        statistics:
            Processing information.
    """

    pages, error = extract_pdf(pdf_path)

    relative_path = str(
        pdf_path.relative_to(PROJECT_ROOT)
    )

    statistics = {
        "file": str(pdf_path),
        "relative_path": relative_path,
        "pages_total": 0,
        "pages_with_text": 0,
        "chunks": 0,
        "status": "failed",
        "error": error,
    }

    if error is not None:

        return [], statistics

    statistics["pages_total"] = len(pages)

    statistics["pages_with_text"] = sum(
        1
        for page in pages
        if page["text"]
    )

    if not pages:

        statistics["status"] = "empty"

        statistics["error"] = (
            "No extractable text found in PDF"
        )

        return [], statistics

    chunks: list[dict[str, Any]] = []

    for page_data in pages:

        page_number = page_data["page"]

        page_text = page_data["text"]

        page_chunks = create_chunks(page_text)

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):

            if len(chunk_text.strip()) < MIN_TEXT_LENGTH:
                continue

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "source_path": str(pdf_path),
                        "relative_path": relative_path,
                        "page": page_number,
                        "chunk": chunk_number,
                    },
                }
            )

    statistics["chunks"] = len(chunks)

    if chunks:

        statistics["status"] = "success"
        statistics["error"] = None

    else:

        statistics["status"] = "empty"

        statistics["error"] = (
            "PDF opened successfully but no usable chunks were produced"
        )

    return chunks, statistics


# ============================================================
# EMBEDDINGS
# ============================================================

def load_embedding_model() -> SentenceTransformer:
    """
    Load sentence-transformers embedding model.
    """

    print()
    print("Loading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")
    print()

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """
    Generate embeddings for all chunks.
    """

    if not texts:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    print()
    print("Generating embeddings...")
    print(f"Texts: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    return embeddings


# ============================================================
# FAISS
# ============================================================

def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """
    Build a FAISS index.

    Since embeddings are normalized, inner product
    is equivalent to cosine similarity.
    """

    if embeddings.ndim != 2:

        raise ValueError(
            f"Expected 2D embeddings, got shape {embeddings.shape}"
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def save_json(
    data: Any,
    output_path: Path,
) -> None:
    """
    Save JSON with UTF-8 encoding.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_vector_store(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
) -> None:
    """
    Save FAISS index and metadata.
    """

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = VECTOR_DIR / "index.faiss"

    metadata_path = VECTOR_DIR / "metadata.json"

    print()
    print("Saving vector store...")

    faiss.write_index(
        index,
        str(index_path),
    )

    save_json(
        metadata,
        metadata_path,
    )

    print(f"FAISS index: {index_path}")
    print(f"Metadata:     {metadata_path}")


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline() -> dict[str, Any]:

    start_time = time.time()

    print()
    print("Starting FBR pipeline...")
    print()

    # --------------------------------------------------------
    # 1. Discover PDFs
    # --------------------------------------------------------

    pdf_files = discover_pdfs()

    pdf_count = len(pdf_files)

    print(
        f"PDFs discovered: {pdf_count}"
    )

    if pdf_count == 0:

        raise RuntimeError(
            f"No PDF files found in:\n{PDF_DIR}"
        )

    # --------------------------------------------------------
    # 2. Process PDFs
    # --------------------------------------------------------

    all_chunks: list[dict[str, Any]] = []

    successful_files: list[str] = []

    empty_files: list[dict[str, Any]] = []

    failed_files: list[dict[str, Any]] = []

    processing_details: list[dict[str, Any]] = []

    total_pages = 0

    pages_with_text = 0

    print()
    print("=" * 70)
    print("PDF PROCESSING")
    print("=" * 70)

    for number, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"[{number}/{pdf_count}] "
            f"{pdf_path.name}"
        )

        chunks, statistics = build_document_chunks(
            pdf_path
        )

        processing_details.append(
            statistics
        )

        total_pages += statistics.get(
            "pages_total",
            0,
        )

        pages_with_text += statistics.get(
            "pages_with_text",
            0,
        )

        status = statistics["status"]

        if status == "success":

            successful_files.append(
                str(pdf_path)
            )

            all_chunks.extend(chunks)

            print(
                f"    ✓ pages={statistics['pages_total']} "
                f"text_pages={statistics['pages_with_text']} "
                f"chunks={statistics['chunks']}"
            )

        elif status == "empty":

            empty_files.append(
                statistics
            )

            print(
                "    ⚠ no usable text"
            )

        else:

            failed_files.append(
                statistics
            )

            print(
                f"    ✗ FAILED: "
                f"{statistics.get('error')}"
            )

    # --------------------------------------------------------
    # 3. Summary before embedding
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOCUMENT PROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"PDFs found:          {pdf_count}"
    )

    print(
        f"PDFs successful:     {len(successful_files)}"
    )

    print(
        f"PDFs empty:          {len(empty_files)}"
    )

    print(
        f"PDFs failed:         {len(failed_files)}"
    )

    print(
        f"Pages discovered:    {total_pages}"
    )

    print(
        f"Pages with text:     {pages_with_text}"
    )

    print(
        f"Chunks:              {len(all_chunks)}"
    )

    # --------------------------------------------------------
    # 4. Stop if no chunks
    # --------------------------------------------------------

    if not all_chunks:

        result = {
            "pdf_count": pdf_count,
            "successful_count": len(
                successful_files
            ),
            "empty_count": len(
                empty_files
            ),
            "failed_count": len(
                failed_files
            ),
            "pages_total": total_pages,
            "pages_with_text": pages_with_text,
            "chunk_count": 0,
            "embedding_count": 0,
            "vector_count": 0,
            "failed_files": failed_files,
            "empty_files": empty_files,
            "processing_details": processing_details,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "status": "failed_no_chunks",
            "elapsed_seconds": round(
                time.time() - start_time,
                2,
            ),
        }

        save_json(
            result,
            VECTOR_DIR / "pipeline_result.json",
        )

        return result

    # --------------------------------------------------------
    # 5. Load embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # 6. Extract texts
    # --------------------------------------------------------

    texts = [
        item["text"]
        for item in all_chunks
    ]

    # --------------------------------------------------------
    # 7. Generate embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        model,
        texts,
    )

    embedding_count = len(
        embeddings
    )

    print()
    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # 8. Validate embeddings
    # --------------------------------------------------------

    if embedding_count != len(
        all_chunks
    ):

        raise RuntimeError(
            "Embedding count does not match chunk count"
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "Embeddings contain NaN or infinite values"
        )

    # --------------------------------------------------------
    # 9. Build FAISS
    # --------------------------------------------------------

    print()
    print("Building FAISS index...")

    index = build_faiss_index(
        embeddings
    )

    vector_count = index.ntotal

    print(
        f"FAISS vectors: {vector_count}"
    )

    # --------------------------------------------------------
    # 10. Save vector store
    # --------------------------------------------------------

    save_vector_store(
        index,
        all_chunks,
    )

    # --------------------------------------------------------
    # 11. Build final result
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    result = {
        "pdf_count": pdf_count,
        "successful_count": len(
            successful_files
        ),
        "empty_count": len(
            empty_files
        ),
        "failed_count": len(
            failed_files
        ),
        "pages_total": total_pages,
        "pages_with_text": pages_with_text,
        "chunk_count": len(
            all_chunks
        ),
        "embedding_count": embedding_count,
        "vector_count": vector_count,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "successful_files": successful_files,
        "empty_files": empty_files,
        "failed_files": failed_files,
        "processing_details": processing_details,
        "vector_store": {
            "index": str(
                VECTOR_DIR / "index.faiss"
            ),
            "metadata": str(
                VECTOR_DIR / "metadata.json"
            ),
        },
        "status": "success",
        "elapsed_seconds": round(
            elapsed,
            2,
        ),
    }

    # --------------------------------------------------------
    # 12. Save pipeline result
    # --------------------------------------------------------

    result_path = (
        VECTOR_DIR
        / "pipeline_result.json"
    )

    save_json(
        result,
        result_path,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print_header()

    try:

        result = run_pipeline()

    except KeyboardInterrupt:

        print()
        print(
            "Pipeline interrupted by user."
        )

        return 130

    except Exception as error:

        print()
        print("=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)
        print(
            f"{type(error).__name__}: {error}"
        )

        logger.exception(
            "Pipeline failed"
        )

        return 1

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE RESULT")
    print("=" * 70)

    print(
        f"PDFs found:       {result['pdf_count']}"
    )

    print(
        f"PDFs successful:   {result['successful_count']}"
    )

    print(
        f"PDFs empty:        {result['empty_count']}"
    )

    print(
        f"PDFs failed:       {result['failed_count']}"
    )

    print(
        f"Pages:             {result['pages_total']}"
    )

    print(
        f"Pages with text:   {result['pages_with_text']}"
    )

    print(
        f"Chunks:            {result['chunk_count']}"
    )

    print(
        f"Embeddings:        {result['embedding_count']}"
    )

    print(
        f"Vectors:           {result['vector_count']}"
    )

    if result.get(
        "embedding_dimension"
    ):

        print(
            f"Dimension:         "
            f"{result['embedding_dimension']}"
        )

    print(
        f"Status:            {result['status']}"
    )

    print(
        f"Time:              "
        f"{result['elapsed_seconds']} seconds"
    )

    # --------------------------------------------------------
    # Failed PDFs
    # --------------------------------------------------------

    if result["failed_count"]:

        print()
        print("=" * 70)
        print(
            "FAILED FILES"
        )
        print("=" * 70)

        for item in result[
            "failed_files"
        ]:

            print(
                f"  - {item['file']}"
            )

            print(
                f"    Error: {item.get('error')}"
            )

    # --------------------------------------------------------
    # Empty PDFs
    # --------------------------------------------------------

    if result["empty_count"]:

        print()
        print("=" * 70)
        print(
            "EMPTY / NON-TEXT FILES"
        )
        print("=" * 70)

        for item in result[
            "empty_files"
        ]:

            print(
                f"  - {item['file']}"
            )

            print(
                f"    Reason: {item.get('error')}"
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("Result saved to:")

    print(
        f"  {VECTOR_DIR / 'pipeline_result.json'}"
    )

    if result["status"] == "success":

        print()
        print(
            "Vector store saved to:"
        )

        print(
            f"  {VECTOR_DIR / 'index.faiss'}"
        )

        print(
            f"  {VECTOR_DIR / 'metadata.json'}"
        )

        print()
        print(
            "FBR vector store build completed successfully."
        )

        return 0

    print()
    print(
        "Vector store was NOT created because "
        "no usable chunks were produced."
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())