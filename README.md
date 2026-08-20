# 🇵🇰 Zabta
## AI-Powered FBR Sales Tax Filing Helper for Pakistani SMEs

Zabta is an AI-assisted Pakistan Federal Board of Revenue (FBR) sales-tax compliance system that combines **Retrieval-Augmented Generation (RAG)**, **BGE-M3 embeddings**, **FAISS vector search**, **deterministic tax-rule resolution**, **date-aware applicability**, and an optional **local Llama 3.1 8B explanation layer through Ollama and LangChain**.

The system is designed to answer a practical question:

> **"Given an invoice item, what FBR sales-tax rate is applicable, which rule/document supports it, and does the invoice's declared tax agree with the resolved rate?"**

The project is built as a hybrid AI system. The language model is **not trusted to invent or independently decide the tax rate**. Instead, FBR evidence is retrieved from the knowledge base, the deterministic resolver classifies and selects the applicable rate, and Llama 3.1 can explain the result using the retrieved evidence.

---

# 🎯 Project Objective

Zabta aims to reduce manual effort in checking sales-tax rates against Pakistan's FBR legislation.

For an invoice, the system can:

1. Read invoice information.
2. Construct an invoice-aware FBR query.
3. Retrieve relevant FBR documents from the local knowledge base.
4. Extract individual tax-rate candidates.
5. Classify rates using their local legal context.
6. Apply invoice-date-aware applicability logic.
7. Select the applicable sales-tax rate.
8. Compare the resolved rate with the invoice's declared GST/sales-tax rate.
9. Recalculate expected sales tax deterministically.
10. Identify rate or tax mismatches.
11. Provide supporting FBR source/page information.
12. Optionally generate a natural-language explanation using local Llama 3.1 through Ollama + LangChain.

---

# 🧠 Core AI Architecture

```text
                         USER / INVOICE
                              │
                              ▼
                    ┌────────────────────┐
                    │  Invoice Parser    │
                    └─────────┬──────────┘
                              │
                              ▼
                    Invoice Information
                              │
                              ▼
                    Query Construction
                              │
                              ▼
                    ┌────────────────────┐
                    │      BGE-M3        │
                    │    Embeddings      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │       FAISS        │
                    │ Vector Retrieval   │
                    └─────────┬──────────┘
                              │
                              ▼
                       FBR Evidence
                              │
                              ▼
                    Tax-Rate Extraction
                              │
                              ▼
                  Rate-Specific Classification
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              Standard     Further      Other
                 │            │         Rates
                 └────────────┴────────────┘
                              │
                              ▼
                    Date-Aware Applicability
                              │
                              ▼
                    Deterministic Tax Rule
                         Resolution
                              │
                              ▼
                    Applicable FBR Rate
                              │
                              ├──────────────► Invoice Tax
                              │                 Validation
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Llama 3.1 8B Instruct │
                 │ Ollama + LangChain     │
                 │ Explanation Layer      │
                 └────────────┬───────────┘
                              │
                              ▼
                    Final Compliance Result
```

---

# 🔥 Why a Hybrid Architecture?

Zabta deliberately separates **retrieval, legal-rule selection, calculation, and explanation**.

### Retrieval

BGE-M3 + FAISS find relevant FBR evidence.

### Deterministic resolution

Python logic determines:

- which rate belongs to which category,
- whether a rate is standard/further/etc.,
- whether the document is relevant to the invoice date,
- which applicable rate should win.

### Calculation

Sales-tax calculations are deterministic Python calculations rather than LLM-generated arithmetic.

### Explanation

Llama 3.1 is used to explain the already-resolved result using the retrieved FBR evidence.

This reduces the risk of an LLM:

- hallucinating a tax rate,
- confusing a standard rate with further tax,
- choosing a future amendment,
- performing incorrect arithmetic,
- inventing a legal citation.

---

# 🏗️ Technical Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Python Version | Python 3.10+ |
| LLM | Llama 3.1 8B Instruct |
| LLM Runtime | Ollama |
| LLM Framework | LangChain |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | FAISS |
| Retrieval | RAG |
| UI | Streamlit |
| Data Processing | Pandas |
| PDF Processing | pdfplumber / project PDF pipeline |
| Tax Resolution | Deterministic Python |
| Testing | pytest |
| Version Control | Git / GitHub |
| Optional Tracking | MLflow |
| Knowledge Base | FBR tax legislation and related documents |

> **Note:** The current implementation uses **BGE-M3**, not the earlier `bge-small-en-v1.5` model.

---

# 📚 Knowledge Base

The knowledge base contains FBR-related tax documents, including material such as:

- Sales Tax Act, 1990
- amendments to the Sales Tax Act
- SROs
- notifications
- rules
- ordinances
- other FBR tax documents

The documents are processed into searchable chunks and indexed for retrieval.

The system uses retrieved evidence rather than relying only on the LLM's pretrained knowledge.

---

# 🔎 RAG Pipeline

The RAG pipeline follows:

```text
FBR PDFs
   │
   ▼
Document Loading
   │
   ▼
Text Extraction
   │
   ▼
Document Processing
   │
   ▼
Chunking
   │
   ▼
BGE-M3 Embeddings
   │
   ▼
FAISS Index
   │
   ▼
Semantic Retrieval
   │
   ▼
FBR Evidence
```

For an invoice query:

```text
Invoice
  │
  ├── Item Description
  ├── HS Code
  ├── Invoice Date
  ├── Purchase Type
  └── Invoice Type
          │
          ▼
Invoice-Aware Query
          │
          ▼
FAISS Retrieval
          │
          ▼
Top-K FBR Evidence
```

---

# 🧾 Invoice Input

The project supports invoice records containing fields such as:

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

Example:

```text
invoice_number: INV-2026-000001
invoice_date: 2025-04-15
seller_name: Tech World Pakistan
seller_ntn: 2345678-9
seller_strn: 3277987654322
buyer_name: Quetta Retailer
buyer_ntn: 1111111-1
item_description: Graphics Card
quantity: 9
unit_price: 85000
taxable_amount: 765000
gst_rate: 0.18
tax_amount: 137700
```

The application can use invoice information to perform both:

1. **FBR rate resolution**
2. **Invoice tax validation**

---

# 📄 Invoice Upload Flow

The intended user flow is:

```text
Upload Invoice
      │
      ▼
Parse Invoice
      │
      ▼
Validate Required Fields
      │
      ▼
For Each Invoice Item
      │
      ▼
Build FBR Query
      │
      ▼
Retrieve FBR Evidence
      │
      ▼
Resolve Applicable Rate
      │
      ▼
Compare With Declared Rate
      │
      ▼
Recalculate Tax
      │
      ▼
VALID / RATE MISMATCH / TAX MISMATCH
```

---

# 🧮 Tax Validation

The system calculates expected sales tax deterministically:

```text
Expected Tax
=
Taxable Amount × Applicable Rate
```

For example:

```text
Taxable Amount = 765,000
Applicable Rate = 18%

Expected Tax
= 765,000 × 0.18
= 137,700
```

The declared invoice tax can then be compared with the expected tax.

---

# 🧠 Tax-Rate Classification

A major part of the project is **rate-specific classification**.

A retrieved FBR chunk can contain multiple rates.

For example:

```text
18% standard sales tax
3% further tax
```

A naive chunk-level classifier could incorrectly classify both rates as further tax.

Zabta instead creates a separate candidate for each extracted rate:

```text
Candidate 1
18%
→ standard

Candidate 2
3%
→ further
```

This is an important design decision in the project.

---

# 🛡️ Zero-Rate Safety

A rate of:

```text
0%
```

is explicitly treated as:

```text
zero-rated
```

and is not allowed to become the standard/base rate.

This prevents a zero-rated provision from being incorrectly selected as the normal sales-tax rate.

---

# 📅 Date-Aware Applicability

Tax rules change over time.

Therefore, Zabta does not simply select the highest semantic retrieval score.

It considers the invoice date and document effective/cutoff information.

Conceptually:

```text
Invoice Date
     │
     ▼
Is document future-dated?
     │
 ┌───┴────┐
Yes       No
 │         │
Reject    Eligible
           │
           ▼
     Prefer latest
     applicable rule
```

For example, if an invoice is dated in 2026:

```text
2022 rate
2023 rate
2024 rate
2025 rate
```

the system should not automatically choose the 2022 result merely because it has a slightly higher retrieval score.

The latest eligible evidence is considered by the applicability logic.

---

# ⚖️ Standard vs Further Tax

Zabta specifically handles the difference between:

### Standard/Base Sales Tax

Example:

```text
18%
```

and:

### Further Tax

Example:

```text
3%
```

or:

```text
4%
```

The system treats them as different applicability categories.

Conceptually:

```text
18% → base
3%  → conditional
4%  → conditional
0%  → conditional / zero-rated
```

This prevents further tax from replacing the base sales-tax rate.

---

# 🧪 Regression Testing

The project uses pytest for regression testing.

Current validated test groups include:

```text
tests/test_invoice.py
tests/test_rate_date_eligibility.py
tests/test_latest_applicable_rate.py
tests/test_invoice_rate_resolver.py
```

The combined test run achieved:

```text
20 passed
```

Command:

```bash
pytest \
tests/test_invoice.py \
tests/test_rate_date_eligibility.py \
tests/test_latest_applicable_rate.py \
tests/test_invoice_rate_resolver.py \
-v
```

Expected result:

```text
20 passed
```

---

# ✅ Tested Capabilities

The tests currently cover areas including:

### Invoice

- invoice loading
- column normalization
- required columns
- required-column validation
- invoice validation
- quality reporting
- sales-tax calculation
- second calculation case
- correct sales-tax comparison
- incorrect sales-tax comparison
- invoice tax dataframe

### Rate Date Eligibility

- newer document preferred for invoice date
- future document rejected
- older document usable when no newer applicable document exists

### Latest Applicable Rate

- latest applicable rate wins for a 2026 invoice
- older 2022 rate does not incorrectly beat a newer applicable rate

### Invoice Rate Resolver

- invoice rate resolution
- standard and further rates classified separately
- 18% is not misclassified as zero-rated
- 18% is not misclassified as further tax when the same chunk contains further tax

---

# 🖥️ Streamlit Demo

The Streamlit application provides a visual demonstration of the pipeline.

The demo can show:

## 1. Invoice Information

```text
Item Description
HS Code
Invoice Date
Purchase Type
Invoice Type
```

## 2. Generated FBR Query

Example:

```text
Determine the applicable Pakistan sales tax rate for the
following invoice item.

Item description: Taxable goods.
HS code: 8471.30.
Invoice date: 2024-05-01.
Purchase type: local purchase.
Invoice type: taxable.
```

## 3. Retrieved FBR Evidence

The interface can display:

- source document
- page
- retrieval score
- retrieved text

## 4. Tax Rate Classification

Example:

```text
18.0% → standard
4.0%  → further
```

## 5. Final Applicable Rate

Example:

```text
Applicable Rate: 18.0%
Category: standard
```

## 6. Invoice Validation

The system compares:

```text
Declared Invoice Rate
          vs
FBR Resolved Rate
```

and:

```text
Declared Tax
          vs
Expected Tax
```

Possible statuses include:

```text
VALID
RATE MISMATCH
TAX MISMATCH
ERROR
```

---

# 🦙 Llama 3.1 + Ollama + LangChain

The current Streamlit integration supports a local Llama 3.1 model.

Recommended model:

```text
llama3.1:8b
```

Ollama runs the model locally.

LangChain provides the interface between the Streamlit application and the Ollama model.

The LLM is used primarily as an **explanation layer**.

### Important design principle

The LLM does not replace the deterministic tax resolver.

Instead:

```text
FBR Evidence
     │
     ▼
Deterministic Resolver
     │
     ├── Applicable Rate
     ├── Category
     ├── Confidence
     └── Source/Page
             │
             ▼
       Llama 3.1
             │
             ▼
       Explanation
```

This allows the project to demonstrate both:

- traditional deterministic engineering
- modern generative AI

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/SamShinwari/Zabta_AI.git
cd Zabta_AI
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
```

## 3. Activate environment

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

If LangChain Ollama support is not already included:

```bash
pip install langchain-ollama
```

---

# 🦙 Ollama Setup

Install Ollama on the system.

Then download the model:

```bash
ollama pull llama3.1:8b
```

Check installed models:

```bash
ollama list
```

Run Ollama:

```bash
ollama serve
```

If Ollama is already running as a service, no additional terminal command may be required.

---

# ▶️ Run Streamlit

From the project root:

```bash
cd ~/Documents/Zabta_AI
source .venv/bin/activate
streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open it in the browser.

---

# 🧪 Run Tests

Run all project tests:

```bash
pytest -v
```

Run the main invoice/rate regression suite:

```bash
pytest \
tests/test_invoice.py \
tests/test_rate_date_eligibility.py \
tests/test_latest_applicable_rate.py \
tests/test_invoice_rate_resolver.py \
-v
```

---

# 📁 Project Structure

A simplified project structure is:

```text
Zabta_AI/
│
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── assets/
│   ├── giki_logo.png
│   └── skylabs_logo.png
│
├── data/
│   ├── raw/
│   │   └── fbr/
│   │       ├── FBR_Acts/
│   │       ├── SROs/
│   │       ├── Notifications/
│   │       ├── rules/
│   │       ├── ordinances/
│   │       └── others/
│   │
│   ├── processed/
│   │
│   └── vector_database/
│       └── fbr/
│
├── src/
│   ├── agent/
│   │
│   ├── fbr/
│   │   ├── citation.py
│   │   ├── current_rate_service.py
│   │   ├── invoice_rate_resolver.py
│   │   ├── rate_applicability.py
│   │   ├── rate_resolver.py
│   │   └── retriever.py
│   │
│   ├── invoice/
│   │   ├── invoice_parser.py
│   │   └── invoice_validation_service.py
│   │
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── report/
│   │
│   ├── tax_engine/
│   │
│   └── utils/
│
├── scripts/
│   ├── preprocess_fbr_docs.py
│   └── build_vector_db.py
│
└── tests/
    ├── test_invoice.py
    ├── test_invoice_rate_candidates.py
    ├── test_invoice_rate_query.py
    ├── test_invoice_rate_resolver.py
    ├── test_invoice_tax_service.py
    ├── test_rate_date_eligibility.py
    └── test_latest_applicable_rate.py
```

---

# 🔐 Citation and Evidence Philosophy

Zabta is designed around evidence-backed tax resolution.

A final result should contain:

```text
Applicable Rate
Category
Confidence
Source
Page
Supporting Text
```

The objective is that a tax-rate claim can be traced back to retrieved FBR evidence.

This is particularly important for a legal/tax domain where unsupported generated answers are risky.

---

# 🧩 Engineering Practices

The project follows several engineering practices:

### Git / GitHub

Development is maintained through Git and GitHub.

The project uses meaningful commits and feature-oriented changes.

### Automated Testing

Important tax-resolution behavior is covered by regression tests.

### Deterministic Tax Calculation

Tax arithmetic is performed using Python logic rather than generated text.

### Evidence-Based Resolution

FBR source and page information are retained with the result.

### Regression Protection

Specific tests protect against previously discovered classification bugs.

---

# 🐛 Important Bug Fixed

One important issue discovered during development was:

```text
18% standard
3% further tax
```

appearing in the same FBR text chunk.

A chunk-level classifier could incorrectly classify:

```text
18%
```

as:

```text
further
```

because the same chunk contained the words "further tax".

The implementation was changed so that classification is performed on the **specific extracted rate occurrence**.

Therefore:

```text
18% → standard
3%  → further
```

This regression is explicitly tested.

Another safety rule ensures:

```text
0% → zero-rated
```

rather than standard.

---

# 📊 Example Resolution

Input:

```text
Item: Taxable goods
HS Code: 8471.30
Invoice Date: 2024-05-01
Purchase Type: local purchase
Invoice Type: taxable
```

Generated query:

```text
Determine the applicable Pakistan sales tax rate for the
following invoice item. Item description: Taxable goods.
HS code: 8471.30. Invoice date: 2024-05-01.
Purchase type: local purchase. Invoice type: taxable.
Identify the applicable sales tax rate, whether it is
standard, reduced, enhanced, zero-rated, exempt, or special,
and provide the supporting FBR document and provision.
```

Retrieved evidence may contain:

```text
18% standard sales tax
3% further tax
4% further tax
```

The resolver separates them:

```text
18% → standard → base
3%  → further → conditional
4%  → further → conditional
```

The date-aware resolver then selects the applicable base rate.

Example final result:

```text
Applicable Rate: 18.0%
Category: standard
Confidence: 89.24%
```

---

# 🎓 Capstone / Viva Demonstration

For a judging/demo session, the recommended flow is:

### Step 1 — Introduce the problem

Explain that manually checking the correct FBR tax rate for invoice items is difficult because:

- FBR documents are numerous,
- rates can change,
- multiple rates may appear in one document,
- standard and further tax can coexist,
- invoice dates matter,
- manual checking is time-consuming.

### Step 2 — Show the invoice

Upload or select an invoice.

Show:

```text
Seller
Buyer
Invoice Date
Item
Quantity
Unit Price
Taxable Amount
Declared GST
Declared Tax
```

### Step 3 — Show the generated query

Explain that the invoice is converted into an FBR-aware retrieval query.

### Step 4 — Show RAG retrieval

Show retrieved FBR documents.

Point out:

```text
Source
Page
Retrieval Score
Evidence
```

### Step 5 — Demonstrate classification

Use the important example:

```text
18% → standard
3% → further
```

Explain that this is **rate-specific classification**, not whole-chunk classification.

### Step 6 — Demonstrate date awareness

Explain:

> "The system does not blindly choose the highest retrieval score. It considers whether the FBR document is applicable to the invoice date."

### Step 7 — Show final rate

For example:

```text
18%
standard
```

### Step 8 — Show invoice validation

Compare:

```text
Invoice declared rate
        vs
FBR applicable rate
```

and:

```text
Invoice declared tax
        vs
System calculated tax
```

### Step 9 — Show Llama explanation

Explain:

> "Llama 3.1 is not deciding the tax rate. It explains the deterministic result using retrieved FBR evidence."

### Step 10 — Show tests

Run:

```bash
pytest \
tests/test_invoice.py \
tests/test_rate_date_eligibility.py \
tests/test_latest_applicable_rate.py \
tests/test_invoice_rate_resolver.py \
-v
```

and show:

```text
20 passed
```

---

# 🏆 Key Project Strengths

Zabta combines:

```text
RAG
+
BGE-M3
+
FAISS
+
FBR Evidence
+
Rate-Specific Classification
+
Date-Aware Applicability
+
Deterministic Tax Calculation
+
Invoice Validation
+
Llama 3.1
+
Ollama
+
LangChain
+
Automated Testing
```

The main engineering principle is:

> **Use retrieval to find the evidence, deterministic logic to make the tax decision, and the LLM to explain the evidence-backed result.**

---

# ⚠️ Limitations

Zabta is a capstone/research prototype and should not be treated as an autonomous legal or tax authority.

Possible limitations include:

- FBR legislation can change after the knowledge base was built.
- Retrieval quality depends on the indexed documents.
- Some special tax regimes may require additional domain-specific rules.
- Legal applicability can depend on conditions not fully represented in a single invoice row.
- The system should be reviewed by a qualified tax professional before being used for real compliance decisions.
- Llama explanations should remain grounded in retrieved evidence.

---

# 🔮 Future Improvements

Potential next steps include:

- full multi-item invoice processing,
- PDF/image invoice OCR,
- stronger HS-code matching,
- more detailed SRO/notification applicability,
- effective-to date handling,
- section/provision extraction,
- citation validation,
- confidence calibration,
- human-in-the-loop review,
- richer compliance reports,
- downloadable validation reports,
- MLflow experiment tracking,
- expanded golden invoice regression datasets,
- production deployment,
- role-based access control,
- audit logs.

---

# 👥 Project

**Zabta — FBR Sales Tax Compliance Assistant**

Built as a two-week AI/ML capstone project.

### Core technologies

```text
Python
BGE-M3
FAISS
RAG
FBR Knowledge Base
Deterministic Tax Rule Resolver
Pandas
Streamlit
Llama 3.1 8B
Ollama
LangChain
pytest
Git / GitHub
```

---

# 📜 Disclaimer

Zabta is an AI-assisted software prototype for educational, research, and demonstration purposes.

It does not replace official FBR guidance, legislation, notifications, professional tax advice, or legal advice.

Always verify important tax decisions against the latest official FBR legislation and applicable notifications.
