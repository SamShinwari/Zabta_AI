from datetime import date

from src.fbr.document_metadata import (
    FBRDocument,
    is_effective_on,
    validate_document,
)


def test_valid_fbr_document():

    document = FBRDocument(
        document_id="FA-2026",
        title="Finance Act 2026",
        document_type="finance_act",
        issue_date=date(2026, 6, 30),
        effective_from=date(2026, 7, 1),
    )

    errors = validate_document(document)

    assert errors == []


def test_invalid_document_type():

    document = FBRDocument(
        document_id="TEST-001",
        title="Test Document",
        document_type="unknown",
        issue_date=date(2026, 6, 30),
        effective_from=date(2026, 7, 1),
    )

    errors = validate_document(document)

    assert len(errors) == 1


def test_effective_document():

    document = FBRDocument(
        document_id="FA-2026",
        title="Finance Act 2026",
        document_type="finance_act",
        issue_date=date(2026, 6, 30),
        effective_from=date(2026, 7, 1),
    )

    assert is_effective_on(
        document,
        date(2026, 8, 15)
    )


def test_document_not_yet_effective():

    document = FBRDocument(
        document_id="FA-2026",
        title="Finance Act 2026",
        document_type="finance_act",
        issue_date=date(2026, 6, 30),
        effective_from=date(2026, 7, 1),
    )

    assert not is_effective_on(
        document,
        date(2026, 6, 30)
    )


def test_expired_document():

    document = FBRDocument(
        document_id="TEST-2025",
        title="Test Document",
        document_type="sro",
        issue_date=date(2025, 6, 20),
        effective_from=date(2025, 7, 1),
        effective_to=date(2026, 6, 30),
    )

    assert not is_effective_on(
        document,
        date(2026, 7, 1)
    )