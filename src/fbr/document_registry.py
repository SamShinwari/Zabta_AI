import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.fbr.document_metadata import (
    FBRDocument,
    validate_document,
)


# ============================================================
# FBR Document Registry
# ============================================================

class FBRDocumentRegistry:
    """
    Registry for FBR legal and tax documents.

    The registry stores document metadata separately from
    the actual PDF/document files.

    Example:

        data/fbr_docs/
            metadata.json

    This registry will later be used by the tax rule engine
    to identify which FBR documents are applicable to an
    invoice date.
    """

    def __init__(
        self,
        metadata_path: str | Path
    ):
        self.metadata_path = Path(
            metadata_path
        )

        self.documents: list[FBRDocument] = []

        self._load()


    # ========================================================
    # Load Registry
    # ========================================================

    def _load(self) -> None:
        """
        Load document metadata from JSON.

        If the file does not exist, an empty registry
        is created.
        """

        if not self.metadata_path.exists():

            self.documents = []

            return

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.documents = []

        for item in data.get(
            "documents",
            []
        ):

            document = self._dict_to_document(
                item
            )

            self.documents.append(
                document
            )


    # ========================================================
    # Save Registry
    # ========================================================

    def save(self) -> None:
        """
        Save all registered FBR documents to JSON.
        """

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "documents": [
                self._document_to_dict(document)
                for document in self.documents
            ]
        }

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )


    # ========================================================
    # Add Document
    # ========================================================

    def add_document(
        self,
        document: FBRDocument
    ) -> None:
        """
        Add a new FBR document to the registry.

        Duplicate document IDs are not allowed.
        """

        errors = validate_document(
            document
        )

        if errors:

            raise ValueError(
                "Invalid FBR document:\n"
                + "\n".join(errors)
            )

        if self.get_document(
            document.document_id
        ) is not None:

            raise ValueError(
                "Document already exists: "
                f"{document.document_id}"
            )

        self.documents.append(
            document
        )

        self.save()


    # ========================================================
    # Get Document
    # ========================================================

    def get_document(
        self,
        document_id: str
    ) -> FBRDocument | None:
        """
        Find an FBR document by document ID.
        """

        for document in self.documents:

            if document.document_id == document_id:

                return document

        return None


    # ========================================================
    # Remove Document
    # ========================================================

    def remove_document(
        self,
        document_id: str
    ) -> bool:
        """
        Remove a document from the registry.

        Returns True if removed, otherwise False.
        """

        original_count = len(
            self.documents
        )

        self.documents = [
            document
            for document in self.documents
            if document.document_id != document_id
        ]

        removed = (
            len(self.documents)
            < original_count
        )

        if removed:

            self.save()

        return removed


    # ========================================================
    # List Documents
    # ========================================================

    def list_documents(
        self
    ) -> list[FBRDocument]:
        """
        Return all registered FBR documents.
        """

        return list(
            self.documents
        )


    # ========================================================
    # Find By Type
    # ========================================================

    def find_by_type(
        self,
        document_type: str
    ) -> list[FBRDocument]:
        """
        Return documents matching a document type.
        """

        document_type = (
            document_type
            .strip()
            .lower()
        )

        return [
            document
            for document in self.documents
            if document.document_type.lower()
            == document_type
        ]


    # ========================================================
    # Find Effective Documents
    # ========================================================

    def find_effective_documents(
        self,
        target_date: date
    ) -> list[FBRDocument]:
        """
        Return documents effective on a particular date.
        """

        results = []

        for document in self.documents:

            if document.effective_from is None:

                continue

            if target_date < document.effective_from:

                continue

            if (
                document.effective_to is not None
                and target_date > document.effective_to
            ):

                continue

            results.append(
                document
            )

        return results


    # ========================================================
    # Convert Document → Dictionary
    # ========================================================

    @staticmethod
    def _document_to_dict(
        document: FBRDocument
    ) -> dict:
        """
        Convert FBRDocument into JSON-compatible dict.
        """

        data = asdict(
            document
        )

        # Convert dates into ISO strings.
        for field_name in [
            "issue_date",
            "effective_from",
            "effective_to",
        ]:

            value = data.get(
                field_name
            )

            if value is not None:

                data[field_name] = (
                    value.isoformat()
                )

        return data


    # ========================================================
    # Convert Dictionary → Document
    # ========================================================

    @staticmethod
    def _dict_to_document(
        data: dict
    ) -> FBRDocument:
        """
        Convert JSON dictionary into FBRDocument.
        """

        date_fields = [
            "issue_date",
            "effective_from",
            "effective_to",
        ]

        converted = dict(
            data
        )

        for field_name in date_fields:

            value = converted.get(
                field_name
            )

            if value:

                converted[field_name] = (
                    date.fromisoformat(value)
                )

        return FBRDocument(
            **converted
        )
