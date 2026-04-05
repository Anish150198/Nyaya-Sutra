"""
Central configuration for Nyaya-Sahayak.
Reads from environment variables (.env for local dev, Databricks App env for prod).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Environment toggle
# ---------------------------------------------------------------------------
NYAYA_ENV = os.getenv("NYAYA_ENV", "local")  # "local" or "databricks"

# ---------------------------------------------------------------------------
# Medallion Architecture — Delta Table Paths
# ---------------------------------------------------------------------------
if NYAYA_ENV == "databricks":
    # Databricks Unity Catalog Volume paths
    NYAYA_DATA_ROOT = os.getenv("NYAYA_DATA_ROOT", "/Volumes/workspace/default/tes/data")
    BRONZE_LAWS_DIR = Path(f"{NYAYA_DATA_ROOT}/bronze/laws")
    BRONZE_DELTA_PATH = f"{NYAYA_DATA_ROOT}/bronze/delta/laws_raw"
    SILVER_DELTA_PATH = f"{NYAYA_DATA_ROOT}/silver/delta/law_chunks"
    GOLD_DELTA_PATH = f"{NYAYA_DATA_ROOT}/gold/delta/law_embeddings"
else:
    # Local paths (relative to project root, using absolute paths for robustness)
    BRONZE_LAWS_DIR = PROJECT_ROOT / "data" / "bronze" / "laws"
    BRONZE_DELTA_PATH = str(PROJECT_ROOT / "data" / "bronze" / "delta" / "laws_raw")
    SILVER_DELTA_PATH = str(PROJECT_ROOT / "data" / "silver" / "delta" / "law_chunks")
    GOLD_DELTA_PATH = str(PROJECT_ROOT / "data" / "gold" / "delta" / "law_embeddings")

# ---------------------------------------------------------------------------
# Acts Registry
# ---------------------------------------------------------------------------
ACTS = {
    "BNS": {
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "data_dir": BRONZE_LAWS_DIR / "BNS" if BRONZE_LAWS_DIR else None,
    },
    "BNSS": {
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "data_dir": BRONZE_LAWS_DIR / "BNSS" if BRONZE_LAWS_DIR else None,
    },
    "BSA": {
        "name": "Bharatiya Sakshya Adhiniyam, 2023",
        "data_dir": BRONZE_LAWS_DIR / "BSA" if BRONZE_LAWS_DIR else None,
    },
    "IPC": {
        "name": "Indian Penal Code, 1860",
        "data_dir": BRONZE_LAWS_DIR / "IPC" if BRONZE_LAWS_DIR else None,
    },
    "CONSTITUTION": {
        "name": "Constitution of India",
        "data_dir": BRONZE_LAWS_DIR / "CONSTITUTION" if BRONZE_LAWS_DIR else None,
    },
    "OTHER_ACTS": {
        "name": "Other State & Central Acts",
        "data_dir": BRONZE_LAWS_DIR / "OTHER_ACTS" if BRONZE_LAWS_DIR else None,
    },
}

# ---------------------------------------------------------------------------
# Databricks
# ---------------------------------------------------------------------------
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_CLUSTER_ID = os.getenv("DATABRICKS_CLUSTER_ID", "")

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "/nyaya-sahayak/evaluation")

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # leave blank for official API

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = 1536  # fixed for text-embedding-3-small

# HuggingFace local embedding (fallback, not used by default)
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Delta table names (Unity Catalog / Hive metastore)
# ---------------------------------------------------------------------------
CATALOG = 'main'
DEFAULT_SCHEMA = 'default'

BRONZE_SCHEMA = f"{CATALOG}.{DEFAULT_SCHEMA}"
SILVER_SCHEMA = f"{CATALOG}.{DEFAULT_SCHEMA}"
GOLD_SCHEMA = f"{CATALOG}.{DEFAULT_SCHEMA}"

TABLE_LAWS_RAW = f"{BRONZE_SCHEMA}.laws_raw"
TABLE_SCHEMES_RAW = f"{BRONZE_SCHEMA}.schemes_raw"
TABLE_IPC_BNS_RAW = f"{BRONZE_SCHEMA}.ipc_bns_raw"

TABLE_LAW_SECTIONS = f"{SILVER_SCHEMA}.law_sections"
TABLE_SCHEMES_CURATED = f"{SILVER_SCHEMA}.schemes_curated"
TABLE_IPC_BNS_MAP = f"{SILVER_SCHEMA}.ipc_bns_map"
TABLE_LEGAL_AID_RULES = f"{SILVER_SCHEMA}.legal_aid_rules"
TABLE_BB_LEGAL_QUESTIONS = f"{SILVER_SCHEMA}.bb_legal_questions"

TABLE_LAW_CHUNKS = f"{GOLD_SCHEMA}.law_chunks"
TABLE_LAW_CHUNKS_EMBEDDED = f"{GOLD_SCHEMA}.law_chunks_embedded"

# ---------------------------------------------------------------------------
# Vector DB
# ---------------------------------------------------------------------------
# Options: "chroma_local" (local default), "chroma_remote", "databricks_vs"
VECTOR_DB = os.getenv("VECTOR_DB", "chroma_local")

# ChromaDB settings
CHROMA_LOCAL_DIR = os.getenv("CHROMA_LOCAL_DIR", str(PROJECT_ROOT / "data" / "gold" / "chromadb"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_AUTH_TOKEN = os.getenv("CHROMA_AUTH_TOKEN", "")
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() == "true"

# Databricks Vector Search
VS_ENDPOINT_NAME = os.getenv("VS_ENDPOINT_NAME", "nyaya_vs_endpoint")
VS_INDEX_NAME = os.getenv("VS_INDEX_NAME", f"{CATALOG}.{DEFAULT_SCHEMA}.gold_vs_index")
VS_SOURCE_TABLE = os.getenv("VS_SOURCE_TABLE", f"{CATALOG}.{DEFAULT_SCHEMA}.gold_law_embeddings")
VS_EMBEDDING_DIM = int(os.getenv("VS_EMBEDDING_DIM", "1536"))

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
