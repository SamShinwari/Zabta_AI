from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FBRQueryAnalysis:
    """
    Analysis of an FBR legal/tax question.

    Extracts explicit legal references from the
    user's question so retrieval can become
    query-aware.
    """

    original_query: str

    sections: list[str]
    rules: list[str]
    sros: list[str]
    years: list[int]

    legal_reference_query: bool

    def has_legal_reference(self) -> bool:
        return self.legal_reference_query


class FBRQueryAnalyzer:
    """
    Analyze FBR questions for legal references.

    Examples:

        "What is section 8B?"
        -> section = 8B

        "What does section 3 of Sales Tax Act say?"
        -> section = 3

        "Explain SRO 1842(I)/2023"
        -> SRO = 1842(I)/2023

        "What was the sales tax rate in 2025?"
        -> year = 2025
    """

    SECTION_PATTERN = re.compile(
        r"\bsection\s+"
        r"(\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))?)",
        re.IGNORECASE,
    )

    RULE_PATTERN = re.compile(
        r"\brule\s+"
        r"(\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))?)",
        re.IGNORECASE,
    )

    SRO_PATTERN = re.compile(
        r"\bS\.?R\.?O\.?\s*"
        r"([0-9]+(?:\([A-Za-z]+\))?/[0-9]{4})",
        re.IGNORECASE,
    )

    YEAR_PATTERN = re.compile(
        r"\b(19\d{2}|20\d{2})\b"
    )

    def analyze(
        self,
        query: str,
    ) -> FBRQueryAnalysis:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        sections = [
            match.upper()
            for match in self.SECTION_PATTERN.findall(
                query
            )
        ]

        rules = [
            match.upper()
            for match in self.RULE_PATTERN.findall(
                query
            )
        ]

        sros = [
            match.upper()
            for match in self.SRO_PATTERN.findall(
                query
            )
        ]

        years = sorted(
            {
                int(match)
                for match in self.YEAR_PATTERN.findall(
                    query
                )
            }
        )

        legal_reference_query = bool(
            sections
            or rules
            or sros
            or years
        )

        return FBRQueryAnalysis(
            original_query=query,
            sections=sections,
            rules=rules,
            sros=sros,
            years=years,
            legal_reference_query=legal_reference_query,
        )