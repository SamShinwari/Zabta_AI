from __future__ import annotations

import streamlit as st

from src.fbr.current_rate_service import FBRCurrentRateService
from src.fbr.invoice_rate_resolver import FBRInvoiceRateResolver


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zabta — FBR Sales Tax Assistant",
    page_icon="🇵🇰",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🇵🇰 Zabta")
st.subheader("AI-Powered FBR Sales Tax & Invoice Validation Assistant")

st.markdown(
    """
    Zabta retrieves relevant **FBR tax legislation** using a RAG pipeline
    and resolves the applicable sales-tax rate for an invoice item.

    **Demo flow:**

    `Invoice Information → FBR Retrieval → Rate Classification → Date Applicability → Result`
    """
)

st.divider()


# ============================================================
# INITIALIZE SERVICES
# ============================================================

@st.cache_resource
def load_services():

    service = FBRCurrentRateService(
        vector_dir="data/vector_database/fbr",
        retrieval_top_k=10,
    )

    resolver = FBRInvoiceRateResolver(
        current_rate_service=service
    )

    return service, resolver


try:
    current_rate_service, invoice_resolver = load_services()

except Exception as exc:

    st.error(
        "Failed to initialize Zabta services."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Zabta")

    st.markdown(
        """
        ### AI Pipeline

        1. Invoice information
        2. Query construction
        3. FBR document retrieval
        4. Tax-rate extraction
        5. Rate classification
        6. Date-aware applicability
        7. Final tax-rate resolution
        """
    )

    st.divider()

    st.caption(
        "Knowledge base: FBR tax documents"
    )

    st.caption(
        "Embedding: BGE-M3"
    )

    st.caption(
        "Vector database: FAISS"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📄 Invoice Item")

col1, col2 = st.columns(2)

with col1:

    item_description = st.text_input(
        "Item Description",
        value="Taxable goods",
    )

    hs_code = st.text_input(
        "HS Code",
        value="8471.30",
    )

    invoice_date = st.date_input(
        "Invoice Date",
        value=None,
    )


with col2:

    purchase_type = st.selectbox(
        "Purchase Type",
        [
            "local purchase",
            "import",
        ],
    )

    invoice_type = st.selectbox(
        "Invoice Type",
        [
            "taxable",
            "zero-rated",
            "exempt",
        ],
    )


st.divider()


# ============================================================
# RESOLVE BUTTON
# ============================================================

if st.button(
    "🔎 Determine Applicable Sales Tax Rate",
    type="primary",
    use_container_width=True,
):

    if not item_description.strip():

        st.error(
            "Please enter an item description."
        )

        st.stop()

    invoice_date_str = (
        invoice_date.isoformat()
        if invoice_date
        else None
    )

    # ========================================================
    # SHOW INPUT
    # ========================================================

    st.header("1️⃣ Invoice Information")

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        st.metric(
            "Item",
            item_description,
        )

    with input_col2:
        st.metric(
            "HS Code",
            hs_code or "Not provided",
        )

    with input_col3:
        st.metric(
            "Invoice Date",
            invoice_date_str or "Not provided",
        )

    # ========================================================
    # BUILD QUERY
    # ========================================================

    query = invoice_resolver.build_query(
        item_description=item_description,
        hs_code=hs_code or None,
        invoice_date=invoice_date_str,
        purchase_type=purchase_type,
        invoice_type=invoice_type,
    )

    with st.expander(
        "🔍 View generated FBR query"
    ):

        st.code(
            query,
            language="text",
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    st.header("2️⃣ FBR Evidence Retrieval")

    with st.spinner(
        "Searching FBR knowledge base..."
    ):

        try:

            results = current_rate_service.retrieve(
                query
            )

        except Exception as exc:

            st.error(
                "FBR retrieval failed."
            )

            st.exception(exc)

            st.stop()

    if not results:

        st.warning(
            "No relevant FBR evidence was retrieved."
        )

        st.stop()

    st.success(
        f"Retrieved {len(results)} relevant FBR evidence chunks."
    )

    # ========================================================
    # SHOW TOP EVIDENCE
    # ========================================================

    with st.expander(
        "📚 View retrieved FBR evidence"
    ):

        for index, result in enumerate(
            results[:5],
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                "N/A",
            )

            score = result.get(
                "score",
                result.get(
                    "retrieval_score",
                    0.0,
                ),
            )

            st.markdown(
                f"### Candidate #{index}"
            )

            st.write(
                f"**Source:** {source}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Retrieval score:** {score:.4f}"
            )

            st.text(
                result.get(
                    "text",
                    "",
                )[:2000]
            )

            st.divider()

    # ========================================================
    # RATE CANDIDATES
    # ========================================================

    st.header("3️⃣ Tax Rate Classification")

    try:

        candidates = (
            invoice_resolver
            ._build_applicability_candidates(
                results,
                invoice_date=invoice_date_str,
            )
        )

    except Exception as exc:

        st.error(
            "Could not build tax-rate candidates."
        )

        st.exception(exc)

        st.stop()

    if not candidates:

        st.warning(
            "No usable tax-rate candidates were found."
        )

        st.stop()

    # ========================================================
    # CANDIDATE TABLE
    # ========================================================

    candidate_rows = []

    for candidate in candidates:

        candidate_rows.append(
            {
                "Rate (%)": candidate.get(
                    "rate"
                ),
                "Category": candidate.get(
                    "category"
                ),
                "Applicability": candidate.get(
                    "applicability"
                ),
                "Year": candidate.get(
                    "year"
                ),
                "Effective From": candidate.get(
                    "effective_from"
                ),
                "Date Relevance": candidate.get(
                    "date_relevance_score"
                ),
                "Retrieval": round(
                    float(
                        candidate.get(
                            "retrieval_score",
                            0.0,
                        )
                    ),
                    4,
                ),
            }
        )

    st.dataframe(
        candidate_rows,
        use_container_width=True,
    )

    # ========================================================
    # FINAL RESOLUTION
    # ========================================================

    st.header("4️⃣ Final Applicable Rate")

    with st.spinner(
        "Resolving applicable sales-tax rate..."
    ):

        try:

            final_result = invoice_resolver.resolve(
                item_description=item_description,
                hs_code=hs_code or None,
                invoice_date=invoice_date_str,
                purchase_type=purchase_type,
                invoice_type=invoice_type,
            )

        except Exception as exc:

            st.error(
                "Rate resolution failed."
            )

            st.exception(exc)

            st.stop()

    # ========================================================
    # RESULT
    # ========================================================

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Applicable Rate",
            f"{final_result.rate}%",
        )

    with result_col2:

        st.metric(
            "Category",
            final_result.category,
        )

    with result_col3:

        st.metric(
            "Confidence",
            f"{final_result.confidence:.2%}",
        )

    st.success(
        "Applicable sales-tax rate successfully resolved."
    )

    # ========================================================
    # SOURCE
    # ========================================================

    st.subheader("📑 Supporting FBR Evidence")

    st.write(
        f"**Source:** {final_result.source}"
    )

    st.write(
        f"**Page:** {final_result.page}"
    )

    if getattr(
        final_result,
        "year",
        None,
    ):

        st.write(
            f"**Document Year:** {final_result.year}"
        )

    if getattr(
        final_result,
        "effective_from",
        None,
    ):

        st.write(
            f"**Effective From:** "
            f"{final_result.effective_from}"
        )

    with st.expander(
        "📖 View supporting FBR text"
    ):

        st.text(
            getattr(
                final_result,
                "text",
                "",
            )
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader("🧠 Resolution Explanation")

    st.markdown(
        f"""
        **Item:** {item_description}

        **HS Code:** {hs_code or "Not provided"}

        **Invoice Date:** {invoice_date_str or "Not provided"}

        **Purchase Type:** {purchase_type}

        **Invoice Type:** {invoice_type}

        **Resolved Rate:** {final_result.rate}%

        **Classification:** {final_result.category}

        The system retrieved FBR evidence, extracted individual
        tax-rate candidates, classified each rate using its local
        legal context, and then applied invoice-date-aware
        applicability ranking.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Zabta — FBR Sales Tax Compliance Assistant | "
    "RAG-based FBR knowledge retrieval"
)