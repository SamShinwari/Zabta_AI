import streamlit as st

from config.logging_config import setup_logging


logger = setup_logging()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Zabta",
    page_icon="📑",
    layout="wide"
)


# ============================================================
# Application
# ============================================================

st.title("📑 Zabta")

st.subheader(
    "AI-Powered FBR Sales Tax Compliance Assistant"
)

st.write(
    """
    Zabta helps SMEs analyze sales invoices,
    determine applicable FBR Sales Tax rules,
    validate tax calculations, and generate
    compliance reports.
    """
)


st.info(
    """
    Phase 0: Project setup

    Invoice processing, dynamic tax-rule resolution,
    FBR RAG, and compliance validation will be
    implemented in subsequent phases.
    """
)