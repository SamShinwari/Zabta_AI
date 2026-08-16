from enum import Enum


class DocumentType(str, Enum):

    SALES_TAX_ACT = "sales_tax_act"

    SALES_TAX_RULES = "sales_tax_rules"

    FINANCE_ACT = "finance_act"

    SRO = "sro"

    NOTIFICATION = "notification"

    CIRCULAR = "circular"

    ORDER = "order"

    OTHER = "other"


class RuleType(str, Enum):

    RATE_CHANGE = "rate_change"

    EXEMPTION = "exemption"

    AMENDMENT = "amendment"

    RESCISSION = "rescission"

    CLARIFICATION = "clarification"

    VALUE_CHANGE = "value_change"

    PROCEDURE = "procedure"

    OTHER = "other"


class ComplianceStatus(str, Enum):

    PASS = "pass"

    RATE_MISMATCH = "rate_mismatch"

    TAX_MISMATCH = "tax_mismatch"

    RULE_NOT_FOUND = "rule_not_found"

    MULTIPLE_RULES = "multiple_rules"

    REVIEW_REQUIRED = "review_required"