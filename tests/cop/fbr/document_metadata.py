from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ============================================================
# FBR Document Types
# ============================================================

DOCUMENT_TYPES = {
    "sales_tax_act",
    "finance_act",
    "sro",
    "notification",
    "circular",
    "general_order",
    "rules",
    "ordinance",
    "other",
}


# ============================================================
# FBR Document Metadata
# ============================================================

@dataclass
class FBRDocument:
    """
    Metadata describing an FBR legal/tax document.

    This class stores document-level information only.

    It does NOT calculate Sales Tax rates.
    """

    document_id: str

    title: str

    document_type: str

    issue_date: date

    effective_from: Optional[date] = None

    effective_to: Optional[date] = None

    source_url: Optional[str] = None

    file_path: Optional[str] = None

    amends_document_id: Optional[str] = None

    replaces_document_id: Optional[str] = None

    description: Optional[str] = None

    tags: list[str] = field(
        default_factory=list
    )


# ============================================================
# Validate Document Type
# ============================================================

def validate_document_type(
    document_type: str
) -> bool:
    """
    Check whether an FBR document type is supported.
    """

    return document_type.lower() in DOCUMENT_TYPES


# ============================================================
# Validate Document Dates
# ============================================================

def validate_document_dates(
    document: FBRDocument
) -> list[str]:
    """
    Validate date relationships inside an FBR document.
    """

    errors = []

    if document.effective_from is not None:

        if document.effective_from < document.issue_date:

            errors.append(
                "effective_from cannot be earlier "
                "than issue_date."
            )

    if (
        document.effective_from is not None
        and document.effective_to is not None
    ):

        if document.effective_to < document.effective_from:

            errors.append(
                "effective_to cannot be earlier "
                "than effective_from."
            )

    return errors


# ============================================================
# Validate Complete Document Metadata
# ============================================================

def validate_document(
    document: FBRDocument
) -> list[str]:
    """
    Validate an FBR document metadata object.
    """

    errors = []

    if not document.document_id:

        errors.append(
            "document_id is required."
        )

    if not document.title:

        errors.append(
            "title is required."
        )

    if not document.document_type:

        errors.append(
            "document_type is required."
        )

    elif not validate_document_type(
        document.document_type
    ):

        errors.append(
            f"Unsupported document type: "
            f"{document.document_type}"
        )

    if document.issue_date is None:

        errors.append(
            "issue_date is required."
        )

    errors.extend(
        validate_document_dates(document)
    )

    return errors


# ============================================================
# Check Whether Document Is Effective
# ============================================================

def is_effective_on(
    document: FBRDocument,
    target_date: date
) -> bool:
    """
    Determine whether an FBR document is effective
    on a particular date.

    Logic:

        target_date >= effective_from

    AND

        target_date <= effective_to

    If effective_to is None, the document is considered
    effective indefinitely after effective_from.
    """

    if document.effective_from is None:

        return False

    if target_date < document.effective_from:

        return False

    if (
        document.effective_to is not None
        and target_date > document.effective_to
    ):

        return False

    return True