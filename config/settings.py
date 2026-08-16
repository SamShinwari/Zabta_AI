from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Project Root
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Main Directories
# ============================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = BASE_DIR / "vector_db"


# ============================================================
# FBR Raw Documents
# ============================================================

FBR_RAW_DIR = RAW_DATA_DIR / "fbr"

FBR_ACTS_DIR = FBR_RAW_DIR / "acts"
FBR_RULES_DIR = FBR_RAW_DIR / "rules"
FBR_FINANCE_ACTS_DIR = FBR_RAW_DIR / "finance_acts"
FBR_SROS_DIR = FBR_RAW_DIR / "sros"
FBR_NOTIFICATIONS_DIR = FBR_RAW_DIR / "notifications"
FBR_CIRCULARS_DIR = FBR_RAW_DIR / "circulars"
FBR_ORDERS_DIR = FBR_RAW_DIR / "orders"
FBR_ACTS_OTHERS = FBR_RAW_DIR / "others"

# ============================================================
# Processed Documents
# ============================================================

TEXT_DATA_DIR = PROCESSED_DATA_DIR / "text"
CLEANED_DATA_DIR = PROCESSED_DATA_DIR / "cleaned"
CHUNKS_DATA_DIR = PROCESSED_DATA_DIR / "chunks"
METADATA_DIR = PROCESSED_DATA_DIR / "metadata"


# ============================================================
# Invoice Data
# ============================================================

INVOICE_DATA_DIR = DATA_DIR / "invoices"

INVOICE_RAW_DIR = INVOICE_DATA_DIR / "raw"
INVOICE_PROCESSED_DIR = INVOICE_DATA_DIR / "processed"
INVOICE_TEST_DIR = INVOICE_DATA_DIR / "test"


# ============================================================
# Tax Rule Database
# ============================================================

TAX_RULES_DIR = DATA_DIR / "tax_rules"

TAX_RULES_FILE = TAX_RULES_DIR / "tax_rules.json"
TAX_RATES_FILE = TAX_RULES_DIR / "tax_rates.json"
EXEMPTIONS_FILE = TAX_RULES_DIR / "exemptions.json"
EFFECTIVE_RULES_FILE = TAX_RULES_DIR / "effective_rules.json"


# ============================================================
# FAISS
# ============================================================

FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss.index"

VECTOR_METADATA_FILE = VECTOR_DB_DIR / "metadata.pkl"

DOCUMENT_STORE_FILE = VECTOR_DB_DIR / "document_store.json"


# ============================================================
# Embedding Model
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3"
)


# ============================================================
# LLM
# ============================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.1:8b"
)


# ============================================================
# RAG Configuration
# ============================================================

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "100")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)