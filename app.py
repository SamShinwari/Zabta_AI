from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.invoice.invoice_parser import InvoiceParser
from src.invoice.invoice_validation_service import InvoiceValidationService
from src.fbr.current_rate_service import FBRCurrentRateService
from src.fbr.invoice_rate_resolver import FBRInvoiceRateResolver

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    OLLAMA_AVAILABLE = True
except ImportError:
    ChatOllama = None
    HumanMessage = None
    OLLAMA_AVAILABLE = False

try:
    from src.fbr.invoice_rate_query import FBRInvoiceRateQuery
except ImportError:
    FBRInvoiceRateQuery = None


st.set_page_config(
    page_title="Zabta — FBR Sales Tax Assistant",
    page_icon="🇵🇰",
    layout="wide",
)


st.title("🇵🇰 Zabta")
st.subheader(
    "AI-Powered FBR Sales Tax Filing Helper for Pakistani SMEs"
)

st.markdown(
    """
Zabta combines **RAG-based FBR document retrieval** with
**deterministic invoice tax validation**.

### AI Pipeline

`Invoice Information → Query Construction → FBR Retrieval → Rate Extraction → Rate Classification → Date Applicability → Final Tax Resolution`
"""
)

st.divider()

st.info(
    "Hybrid AI: BGE-M3 + FAISS retrieve FBR evidence; "
    "the deterministic Python resolver decides the rate; "
    "Llama 3.1 via Ollama + LangChain generates the explanation."
)


@st.cache_resource
def initialize_llm(model_name: str):
    if not OLLAMA_AVAILABLE:
        return None
    return ChatOllama(
        model=model_name,
        temperature=0,
    )


def generate_llm_explanation(
    llm,
    item_description,
    hs_code,
    invoice_date,
    purchase_type,
    invoice_type,
    resolved_rate,
    category,
    confidence,
    source,
    page,
    evidence_text,
):
    prompt = f"""
You are the explanation layer of Zabta, an FBR sales-tax
compliance assistant.

The deterministic Python tax resolver has already selected
the applicable rate. Do NOT change, invent, or recalculate it.

Invoice:
Item: {item_description}
HS Code: {hs_code or "Not provided"}
Invoice date: {invoice_date}
Purchase type: {purchase_type}
Invoice type: {invoice_type}

Deterministic result:
Applicable rate: {resolved_rate}%
Category: {category}
Confidence: {confidence}
FBR source: {source}
Page: {page}

Retrieved FBR evidence:
{evidence_text}

Give a concise explanation suitable for a viva/demo dashboard.
Explain why the evidence supports the classification and why
the invoice date matters. Mention the source and page.
Use only the supplied evidence. Do not invent legal provisions.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(x) for x in content)
    return str(content)


with st.sidebar:
    st.header("⚙️ System Configuration")

    vector_dir = st.text_input(
        "FAISS Vector Database",
        value="data/vector_database/fbr",
    )

    retrieval_top_k = st.slider(
        "FBR Evidence Top-K",
        min_value=5,
        max_value=20,
        value=10,
    )

    st.divider()

    st.header("🧠 Local LLM")
    ollama_model = st.text_input(
        "Ollama Model",
        value="llama3.1:8b",
    )
    use_ollama = st.checkbox(
        "Enable Llama 3.1 explanation",
        value=True,
        disabled=not OLLAMA_AVAILABLE,
    )

    if not OLLAMA_AVAILABLE:
        st.warning(
            "Install LangChain Ollama with: "
            "pip install langchain-ollama"
        )

    st.divider()

    st.markdown(
        """
### Technical Stack

- **LLM:** Llama 3.1 8B Instruct
- **Embeddings:** BGE-M3
- **Vector Database:** FAISS
- **RAG:** FBR legislation retrieval
- **Framework:** LangChain
- **UI:** Streamlit
- **Data:** Pandas
- **Tax Engine:** Python
- **Testing:** Pytest
"""
    )

    st.divider()
    st.caption("Zabta — FBR Sales Tax Compliance Assistant")


@st.cache_resource
def initialize_services(vector_directory: str, top_k: int):
    current_rate_service = FBRCurrentRateService(
        vector_dir=vector_directory,
        retrieval_top_k=top_k,
    )

    validation_service = InvoiceValidationService(
        current_rate_service=current_rate_service,
    )

    parser = InvoiceParser()

    return parser, current_rate_service, validation_service


try:
    parser, current_rate_service, validation_service = (
        initialize_services(vector_dir, retrieval_top_k)
    )
    invoice_rate_resolver = FBRInvoiceRateResolver(
        current_rate_service=current_rate_service
    )
    llm = initialize_llm(ollama_model) if use_ollama else None
except Exception as exc:
    st.error("❌ Could not initialize Zabta.")
    st.exception(exc)
    st.stop()


st.header("📄 Upload Invoice")

uploaded_file = st.file_uploader(
    "Upload your invoice CSV file",
    type=["csv", "tsv", "txt"],
    help="Upload an invoice file containing invoice and sales-tax information.",
)

if uploaded_file is None:
    st.info("👆 Upload an invoice CSV file to start.")

    st.markdown(
        """
### Expected Invoice Columns

```text
invoice_number
invoice_date
seller_name
seller_ntn
seller_strn
buyer_name
buyer_ntn
item_description
quantity
unit_price
taxable_amount
gst_rate
tax_amount
```

### Example

```text
INV-2026-000001
2025-04-15
Tech World Pakistan
Graphics Card
9
85000
765000
0.18
137700
```
"""
    )

    st.stop()


try:
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        temporary_path = temporary_file.name

except Exception as exc:
    st.error("❌ Could not process uploaded file.")
    st.exception(exc)
    st.stop()


try:
    df = parser.load(temporary_path)
except Exception as exc:
    st.error("❌ Invoice could not be loaded.")
    st.exception(exc)
    st.stop()


st.header("1️⃣ Invoice Information")

st.success(f"Successfully loaded **{len(df)} invoice row(s)**.")

st.dataframe(
    df,
    use_container_width=True,
)


st.header("2️⃣ Invoice Structure Validation")

validation_reports = []

for _, row in df.iterrows():
    try:
        report = parser.validate_invoice(row)
        validation_reports.append(report)
    except Exception as exc:
        validation_reports.append(
            {
                "errors": [str(exc)],
                "warnings": [],
            }
        )


total_errors = sum(
    len(report.get("errors", []))
    for report in validation_reports
)

total_warnings = sum(
    len(report.get("warnings", []))
    for report in validation_reports
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Invoice Rows", len(df))

with col2:
    st.metric("Errors", total_errors)

with col3:
    st.metric("Warnings", total_warnings)


if total_errors == 0:
    st.success("✅ Invoice structure validation passed.")
else:
    st.warning(f"⚠️ {total_errors} validation error(s) detected.")


st.header("3️⃣ FBR Query Construction")

first_row = df.iloc[0]


def build_demo_query(row):
    item = str(row.get("item_description", ""))

    hs_code = row.get("hs_code", None)

    if pd.isna(hs_code):
        hs_code = None
    elif hs_code is not None:
        hs_code = str(hs_code)

    invoice_date = str(row.get("invoice_date", ""))

    if FBRInvoiceRateQuery is not None:
        try:
            return FBRInvoiceRateQuery.build_query(
                item_description=item,
                hs_code=hs_code,
                invoice_date=invoice_date,
                purchase_type="local purchase",
                invoice_type="taxable",
            )
        except Exception:
            pass

    return (
        "Determine the applicable Pakistan sales tax rate "
        "for the following invoice item. "
        f"Item description: {item}. "
        f"HS code: {hs_code or 'Not provided'}. "
        f"Invoice date: {invoice_date}. "
        "Purchase type: local purchase. "
        "Invoice type: taxable. "
        "Identify the applicable sales tax rate, "
        "classification, supporting FBR document and provision."
    )


generated_query = build_demo_query(first_row)

with st.expander("🔍 View generated FBR query"):
    st.code(generated_query, language="text")


st.header("4️⃣ FBR Evidence Retrieval")

show_evidence = st.checkbox(
    "📚 Retrieve and view FBR evidence",
    value=False,
)


if show_evidence:
    try:
        evidence = current_rate_service.retrieve(generated_query)

        st.success(
            f"Retrieved {len(evidence)} relevant FBR evidence chunk(s)."
        )

        for index, result in enumerate(evidence, start=1):
            metadata = result.get("metadata", {})

            source = metadata.get(
                "source",
                "Unknown FBR document",
            )

            page = metadata.get("page", None)

            score = result.get(
                "score",
                result.get("retrieval_score", 0.0),
            )

            with st.expander(f"📚 Candidate #{index}"):
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Page:** {page}")
                st.markdown(
                    f"**Retrieval score:** {float(score):.4f}"
                )
                st.text(result.get("text", ""))

    except Exception as exc:
        st.error("Could not retrieve FBR evidence.")
        st.exception(exc)


st.header("5️⃣ FBR Sales Tax Validation")

run_validation = st.button(
    "🔍 Validate Invoice Against FBR",
    type="primary",
    use_container_width=True,
)


if run_validation:
    results = []
    progress = st.progress(0)
    status_text = st.empty()

    for index, row in df.iterrows():
        status_text.write(
            f"Processing invoice {index + 1} of {len(df)}..."
        )

        invoice_number = row.get("invoice_number", "")
        item_description = str(row.get("item_description", ""))
        hs_code = row.get("hs_code", None)
        if pd.isna(hs_code):
            hs_code = None
        elif hs_code is not None:
            hs_code = str(hs_code)

        invoice_date = str(row.get("invoice_date", ""))
        purchase_type = str(row.get("purchase_type", "local purchase"))
        invoice_type = str(row.get("invoice_type", "taxable"))

        try:
            query = FBRInvoiceRateResolver.build_query(
                item_description=item_description,
                hs_code=hs_code,
                invoice_date=invoice_date,
                purchase_type=purchase_type,
                invoice_type=invoice_type,
            )

            evidence = current_rate_service.retrieve(query)
            if not evidence:
                raise LookupError("No FBR evidence was retrieved.")

            resolution = invoice_rate_resolver.resolve(
                item_description=item_description,
                hs_code=hs_code,
                invoice_date=invoice_date,
                purchase_type=purchase_type,
                invoice_type=invoice_type,
            )

            resolved_rate = getattr(resolution, "rate", None)
            if resolved_rate is None:
                resolved_rate = getattr(
                    resolution,
                    "applicable_rate",
                    None,
                )

            category = getattr(resolution, "category", "unknown")
            confidence = getattr(resolution, "confidence", 0.0)
            source = getattr(resolution, "source", "")
            page = getattr(resolution, "page", None)

            taxable_amount = float(row.get("taxable_amount", 0) or 0)
            declared_tax = float(row.get("tax_amount", 0) or 0)

            declared_rate = row.get("gst_rate", None)
            declared_rate_percent = None
            if pd.notna(declared_rate):
                declared_rate_percent = float(declared_rate)
                if declared_rate_percent <= 1:
                    declared_rate_percent *= 100

            expected_tax = (
                taxable_amount * float(resolved_rate) / 100
                if resolved_rate is not None
                else None
            )

            rate_match = (
                declared_rate_percent is not None
                and resolved_rate is not None
                and abs(
                    declared_rate_percent - float(resolved_rate)
                ) < 0.0001
            )

            tax_match = (
                expected_tax is not None
                and abs(declared_tax - expected_tax) < 0.01
            )

            if not rate_match:
                status = "RATE MISMATCH"
            elif not tax_match:
                status = "TAX MISMATCH"
            else:
                status = "VALID"

            supporting_text = ""
            for item in evidence:
                meta = item.get("metadata", {})
                if (
                    meta.get("source", "") == source
                    and meta.get("page", None) == page
                ):
                    supporting_text = item.get("text", "")
                    break

            if not supporting_text and evidence:
                supporting_text = evidence[0].get("text", "")

            ai_explanation = ""
            if llm is not None:
                try:
                    ai_explanation = generate_llm_explanation(
                        llm,
                        item_description,
                        hs_code,
                        invoice_date,
                        purchase_type,
                        invoice_type,
                        resolved_rate,
                        category,
                        confidence,
                        source,
                        page,
                        supporting_text,
                    )
                except Exception as exc:
                    ai_explanation = (
                        f"Llama explanation unavailable: {exc}"
                    )

            results.append({
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "item_description": item_description,
                "declared_rate": declared_rate_percent,
                "applicable_rate": resolved_rate,
                "taxable_amount": taxable_amount,
                "declared_tax": declared_tax,
                "expected_tax": expected_tax,
                "rate_match": rate_match,
                "tax_match": tax_match,
                "category": category,
                "confidence": confidence,
                "source": source,
                "page": page,
                "status": status,
                "explanation": ai_explanation,
                "query": query,
                "supporting_text": supporting_text,
            })

        except Exception as exc:
            results.append({
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "item_description": item_description,
                "declared_rate": None,
                "applicable_rate": None,
                "taxable_amount": row.get("taxable_amount", 0),
                "declared_tax": row.get("tax_amount", 0),
                "expected_tax": None,
                "rate_match": False,
                "tax_match": False,
                "category": "error",
                "confidence": 0.0,
                "source": "",
                "page": None,
                "status": "ERROR",
                "explanation": str(exc),
                "query": "",
                "supporting_text": "",
            })

        progress.progress((index + 1) / len(df))

    status_text.empty()
    progress.empty()
    st.session_state["validation_results"] = pd.DataFrame(results)


if "validation_results" in st.session_state:
    validation_df = st.session_state["validation_results"]

    st.header("6️⃣ Final Validation Results")

    status_series = (
        validation_df["status"]
        .astype(str)
        .str.upper()
    )

    valid_count = int(
        (status_series == "VALID").sum()
    )

    rate_mismatch_count = int(
        (status_series == "RATE MISMATCH").sum()
    )

    tax_mismatch_count = int(
        (status_series == "TAX MISMATCH").sum()
    )

    error_count = int(
        (status_series == "ERROR").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("✅ Valid", valid_count)

    with col2:
        st.metric("⚠️ Rate Mismatch", rate_mismatch_count)

    with col3:
        st.metric("❌ Tax Mismatch", tax_mismatch_count)

    with col4:
        st.metric("🚨 Errors", error_count)

    st.subheader("Invoice Validation Summary")

    display_columns = [
        "invoice_number",
        "item_description",
        "declared_rate",
        "applicable_rate",
        "declared_tax",
        "expected_tax",
        "category",
        "confidence",
        "status",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in validation_df.columns
    ]

    st.dataframe(
        validation_df[available_columns],
        use_container_width=True,
    )

    st.subheader("🔎 Invoice-Level Analysis")

    invoice_numbers = (
        validation_df["invoice_number"]
        .astype(str)
        .tolist()
    )

    if invoice_numbers:
        selected_invoice = st.selectbox(
            "Select an invoice",
            invoice_numbers,
        )

        selected_rows = validation_df[
            validation_df["invoice_number"]
            .astype(str)
            == selected_invoice
        ]

        if not selected_rows.empty:
            selected = selected_rows.iloc[0]

            selected_status = str(
                selected.get("status", "UNKNOWN")
            ).upper()

            if selected_status == "VALID":
                st.success(
                    f"✅ Invoice {selected_invoice} is VALID"
                )

            elif selected_status == "RATE MISMATCH":
                st.warning(
                    f"⚠️ Invoice {selected_invoice} has a RATE MISMATCH"
                )

            elif selected_status == "TAX MISMATCH":
                st.error(
                    f"❌ Invoice {selected_invoice} has a TAX MISMATCH"
                )

            elif selected_status == "ERROR":
                st.error(
                    f"🚨 Invoice {selected_invoice} could not be resolved"
                )

            else:
                st.info(f"Status: {selected_status}")

            left, right = st.columns(2)

            with left:
                st.markdown("### 📄 Invoice Information")

                st.write(
                    "**Invoice Number:**",
                    selected.get("invoice_number", ""),
                )

                st.write(
                    "**Invoice Date:**",
                    selected.get("invoice_date", ""),
                )

                st.write(
                    "**Item:**",
                    selected.get("item_description", ""),
                )

                taxable_amount = selected.get(
                    "taxable_amount",
                    0,
                )

                declared_tax = selected.get(
                    "declared_tax",
                    0,
                )

                declared_rate = selected.get(
                    "declared_rate",
                    None,
                )

                try:
                    st.write(
                        "**Taxable Amount:**",
                        f"{float(taxable_amount):,.2f}",
                    )
                except (TypeError, ValueError):
                    st.write(
                        "**Taxable Amount:**",
                        taxable_amount,
                    )

                try:
                    declared_rate_value = float(
                        declared_rate
                    )

                    # Your CSV stores 0.18,
                    # while the resolver may return 18.
                    if declared_rate_value <= 1:
                        declared_rate_value *= 100

                    st.write(
                        "**Declared Rate:**",
                        f"{declared_rate_value:.2f}%",
                    )
                except (TypeError, ValueError):
                    st.write(
                        "**Declared Rate:**",
                        declared_rate,
                    )

                try:
                    st.write(
                        "**Declared Tax:**",
                        f"{float(declared_tax):,.2f}",
                    )
                except (TypeError, ValueError):
                    st.write(
                        "**Declared Tax:**",
                        declared_tax,
                    )

            with right:
                st.markdown("### 📚 FBR Resolution")

                applicable_rate = selected.get(
                    "applicable_rate",
                    None,
                )

                if pd.notna(applicable_rate):
                    try:
                        applicable_rate_display = (
                            f"{float(applicable_rate):.2f}%"
                        )
                    except (TypeError, ValueError):
                        applicable_rate_display = str(
                            applicable_rate
                        )
                else:
                    applicable_rate_display = "N/A"

                st.write(
                    "**Applicable Rate:**",
                    applicable_rate_display,
                )

                st.write(
                    "**Category:**",
                    selected.get(
                        "category",
                        "unknown",
                    ),
                )

                expected_tax = selected.get(
                    "expected_tax",
                    None,
                )

                if pd.notna(expected_tax):
                    try:
                        expected_tax_display = (
                            f"{float(expected_tax):,.2f}"
                        )
                    except (TypeError, ValueError):
                        expected_tax_display = str(
                            expected_tax
                        )
                else:
                    expected_tax_display = "N/A"

                st.write(
                    "**Expected Tax:**",
                    expected_tax_display,
                )

                confidence = selected.get(
                    "confidence",
                    0.0,
                )

                try:
                    confidence_value = float(
                        confidence
                    )

                    if confidence_value <= 1:
                        confidence_value *= 100

                    st.write(
                        "**Confidence:**",
                        f"{confidence_value:.2f}%",
                    )
                except (TypeError, ValueError):
                    st.write(
                        "**Confidence:**",
                        confidence,
                    )

                st.write(
                    "**FBR Source:**",
                    selected.get(
                        "source",
                        "Unknown",
                    ),
                )

                st.write(
                    "**Page:**",
                    selected.get(
                        "page",
                        "N/A",
                    ),
                )

            st.subheader("🧠 Resolution Explanation")

            explanation = selected.get(
                "explanation",
                "",
            )

            if explanation:
                st.info(explanation)
            else:
                st.info(
                    """
The system retrieved FBR evidence, extracted
individual tax-rate candidates, classified each
rate using its local legal context, applied
invoice-date-aware applicability ranking, and
compared the resolved FBR rate with the invoice rate.
"""
                )

            st.subheader("📑 Supporting FBR Evidence")

            source = selected.get("source", "")
            page = selected.get("page", None)

            if source:
                st.markdown(f"**Source:** {source}")

            if pd.notna(page):
                st.markdown(f"**Page:** {page}")

            show_supporting_text = st.checkbox(
                "📖 View supporting FBR text",
                key=f"supporting_{selected_invoice}",
            )

            if show_supporting_text:
                supporting_text = selected.get(
                    "supporting_text",
                    "",
                )
                if supporting_text:
                    st.text(supporting_text)
                else:
                    st.info(
                        "No supporting FBR text was stored for this result."
                    )

    st.divider()

    st.subheader("📥 Export Results")

    csv_data = validation_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Validation Report",
        data=csv_data,
        file_name="zabta_invoice_validation.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()

st.caption(
    "Zabta — FBR Sales Tax Compliance Assistant | "
    "RAG-based FBR knowledge retrieval"
)
