from datetime import date

import pytest

from src.fbr.document_metadata import FBRDocument
from src.fbr.document_registry import FBRDocumentRegistry


# ============================================================
# Test Document Factory
# ============================================================

def create_test_document():

    return FBRDocument(
        document_id="FA-2026-TEST",
        title="Finance Act 2026 Test",
        document_type="finance_act",
        issue_date=date(
            2026,
            6,
            30
        ),
        effective_from=date(
            2026,
            7,
            1
        ),
        source_url="https://example.com",
        description="Test document",
        tags=[
            "sales_tax",
            "finance_act",
        ],
    )


# ============================================================
# Test Registry Fixture
# ============================================================

@pytest.fixture
def registry(tmp_path):
    """
    Create a completely fresh registry for every test.

    pytest's tmp_path gives each test its own temporary
    directory, so tests cannot interfere with each other.
    """

    metadata_path = (
        tmp_path / "metadata.json"
    )

    return FBRDocumentRegistry(
        metadata_path
    )


# ============================================================
# Test: Add Document
# ============================================================

def test_add_document(registry):

    document = create_test_document()

    registry.add_document(
        document
    )

    result = registry.get_document(
        "FA-2026-TEST"
    )

    assert result is not None

    assert result.title == (
        "Finance Act 2026 Test"
    )


# ============================================================
# Test: Registry Persistence
# ============================================================

def test_registry_persistence(
    registry
):

    document = create_test_document()

    registry.add_document(
        document
    )

    # Create a second registry using
    # the same metadata file.
    new_registry = FBRDocumentRegistry(
        registry.metadata_path
    )

    result = new_registry.get_document(
        "FA-2026-TEST"
    )

    assert result is not None

    assert result.document_id == (
        "FA-2026-TEST"
    )

    assert result.title == (
        "Finance Act 2026 Test"
    )


# ============================================================
# Test: Find By Type
# ============================================================

def test_find_by_type(registry):

    document = create_test_document()

    registry.add_document(
        document
    )

    results = registry.find_by_type(
        "finance_act"
    )

    assert len(results) == 1

    assert results[0].document_id == (
        "FA-2026-TEST"
    )


# ============================================================
# Test: Find Effective Documents
# ============================================================

def test_find_effective_documents(
    registry
):

    document = create_test_document()

    registry.add_document(
        document
    )

    results = registry.find_effective_documents(
        date(
            2026,
            8,
            15
        )
    )

    assert len(results) == 1

    assert results[0].document_id == (
        "FA-2026-TEST"
    )


# ============================================================
# Test: Document Not Yet Effective
# ============================================================

def test_document_not_effective_before_date(
    registry
):

    document = create_test_document()

    registry.add_document(
        document
    )

    results = registry.find_effective_documents(
        date(
            2026,
            6,
            30
        )
    )

    assert len(results) == 0


# ============================================================
# Test: Expired Document
# ============================================================

def test_expired_document(
    registry
):

    document = FBRDocument(
        document_id="SRO-2025-TEST",
        title="Test SRO",
        document_type="sro",
        issue_date=date(
            2025,
            6,
            20
        ),
        effective_from=date(
            2025,
            7,
            1
        ),
        effective_to=date(
            2026,
            6,
            30
        ),
    )

    registry.add_document(
        document
    )

    results = registry.find_effective_documents(
        date(
            2026,
            7,
            1
        )
    )

    assert len(results) == 0


# ============================================================
# Test: Duplicate Document
# ============================================================

def test_duplicate_document_rejected(
    registry
):

    document = create_test_document()

    registry.add_document(
        document
    )

    with pytest.raises(
        ValueError,
        match="Document already exists"
    ):

        registry.add_document(
            document
        )


# ============================================================
# Test: Remove Document
# ============================================================

def test_remove_document(
    registry
):

    document = create_test_document()

    registry.add_document(
        document
    )

    removed = registry.remove_document(
        "FA-2026-TEST"
    )

    assert removed is True

    result = registry.get_document(
        "FA-2026-TEST"
    )

    assert result is None