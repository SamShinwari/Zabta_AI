import streamlit as st

from config.logging_config import setup_logging
from src.fbr.service import ZabtaFBRService


logger = setup_logging()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Zabta",
    page_icon="📑",
    layout="wide",
)


# ============================================================
# FBR Service
# ============================================================

@st.cache_resource
def get_fbr_service():
    """
    Initialize the FBR service once and reuse it
    across Streamlit reruns.
    """

    return ZabtaFBRService(
        model="llama3.1:8b",
        retrieval_top_k=10,
        final_top_k=5,
        temperature=0,
    )


# ============================================================
# Header
# ============================================================

st.title("📑 Zabta")

st.subheader(
    "AI-Powered FBR Sales Tax Compliance Assistant"
)

st.write(
    """
    Zabta helps Pakistani SMEs understand FBR sales tax
    requirements using official FBR documents and
    retrieval-augmented generation.
    """
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Zabta Configuration")

    st.write(
        "**LLM:** Llama 3.1 8B"
    )

    st.write(
        "**Backend:** Ollama"
    )

    st.write(
        "**Embedding:** BAAI/bge-m3"
    )

    st.write(
        "**Vector Database:** FAISS"
    )

    st.write(
        "**Retrieved documents:** 10"
    )

    st.write(
        "**Final sources:** 5"
    )


# ============================================================
# Initialize Service
# ============================================================

try:

    service = get_fbr_service()

except Exception as exc:

    logger.exception(
        "Failed to initialize Zabta FBR service"
    )

    st.error(
        "Unable to initialize the FBR service."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# FBR Question Answering
# ============================================================

st.header("💬 Ask Zabta")

question = st.text_area(
    "Ask an FBR sales tax question:",
    placeholder=(
        "Example: What is the standard sales tax "
        "rate in Pakistan?"
    ),
    height=120,
)


ask_button = st.button(
    "🔎 Ask Zabta",
    type="primary",
)


# ============================================================
# Process Question
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter an FBR question."
        )

        st.stop()

    with st.spinner(
        "Searching FBR documents and generating answer..."
    ):

        try:

            response = service.ask(
                question
            )

        except Exception as exc:

            logger.exception(
                "FBR question failed"
            )

            st.error(
                "Zabta could not process the question."
            )

            st.exception(exc)

            st.stop()

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    st.header("📌 Answer")

    st.write(
        response.answer
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    if response.sources:

        st.header("📚 Sources")

        for number, source in enumerate(
            response.sources,
            start=1,
        ):

            citation = source.get(
                "citation",
                "Unknown source",
            )

            with st.expander(
                f"[{number}] {citation}"
            ):

                st.write(
                    f"**Source:** "
                    f"{source.get('source', citation)}"
                )

                if source.get("page") is not None:

                    st.write(
                        f"**Page:** "
                        f"{source.get('page')}"
                    )

                if source.get("chunk") is not None:

                    st.write(
                        f"**Chunk:** "
                        f"{source.get('chunk')}"
                    )

    # --------------------------------------------------------
    # Retrieval Statistics
    # --------------------------------------------------------

    with st.expander(
        "🔧 Retrieval Information"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Retrieved",
                response.retrieved_count,
            )

        with col2:

            st.metric(
                "Reranked",
                response.reranked_count,
            )


# ============================================================
# Footer / System Information
# ============================================================

with st.expander(
    "ℹ️ System Information"
):

    try:

        information = service.info()

        st.json(
            information
        )

    except Exception as exc:

        st.warning(
            f"Could not load system information: {exc}"
        )