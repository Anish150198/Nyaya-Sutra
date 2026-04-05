# Nyaya-Sahayak — Medallion Pipeline Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (offline)                       │
│                                                                 │
│  Scraped JSON    ──►  BRONZE Delta    ──►  SILVER Delta         │
│  (raw files)         (laws_raw)           (law_chunks)          │
│                                                                 │
│                                        ──►  GOLD Delta          │
│                                             (law_embeddings)    │
│                                                  │              │
│                                                  ▼              │
│                                           ChromaDB / FAISS      │
│                                           (vector index)        │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING (online)                              │
│                                                                 │
│  User query → Retriever → LLM (OpenAI) → Streamlit UI          │
│                 │                                               │
│                 └── smart section extraction                     │
│                 └── act auto-detection                          │
│                 └── hybrid: exact lookup + semantic search       │
└─────────────────────────────────────────────────────────────────┘
```

### Medallion Layers

| Layer | What | Format | Path (local) |
|-------|------|--------|--------------|
| **Bronze** | Raw scraped sections (1 row per section) | Delta | `data/bronze/delta/laws_raw/` |
| **Silver** | Chunked text (~800 chars, 150 overlap) | Delta | `data/silver/delta/law_chunks/` |
| **Gold** | Embeddings (1536-dim OpenAI vectors) + text | Delta | `data/gold/delta/law_embeddings/` |
| **Vector DB** | ChromaDB index for fast similarity search | ChromaDB | `data/gold/chromadb/` |

---

## Quick Start (Local)

### Prerequisites

```bash
# Python 3.10+ with conda
conda activate nb

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Scrape Law Data (if not already done)

Downloads BNS, BNSS, BSA sections from advocatekhoj.com:

```bash
python scripts/scrape_acts.py
```

This creates JSON files in `data/bronze/laws/{BNS,BNSS,BSA}/json/`.
Takes ~15 minutes (rate-limited to 1 req/sec).

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### Step 3: Run the Full Medallion Pipeline

**Option A — Medallion mode (recommended):**

```bash
# All three stages in one command
python -m pipelines.run_all

# Or run individual stages:
python -m pipelines.run_all --stage bronze   # JSON → Bronze Delta
python -m pipelines.run_all --stage silver   # Bronze → Silver Delta (chunking)
python -m pipelines.run_all --stage gold     # Silver → Gold Delta + ChromaDB
```

**Option B — Legacy mode (skips Delta, direct to vector DB):**

```bash
python build_indexes.py
```

**Option C — Medallion via build_indexes:**

```bash
python build_indexes.py --medallion
```

### Step 4: Launch the UI

```bash
streamlit run app/main.py
```

---

## Running on Databricks (Community Edition — Free)

### Step 1: Create a Databricks Account

1. Go to https://community.cloud.databricks.com
2. Sign up for the free Community Edition
3. Create a cluster (any default config works)

### Step 2: Upload Data to DBFS

Upload the scraped JSON folders to DBFS:

```
/FileStore/nyaya/bronze/laws/BNS/json/*.json
/FileStore/nyaya/bronze/laws/BNSS/json/*.json
/FileStore/nyaya/bronze/laws/BSA/json/*.json
```

You can do this via:
- **Databricks UI**: Data → DBFS → Upload
- **Databricks CLI**:
  ```bash
  pip install databricks-cli
  databricks configure --token
  # Enter host: https://community.cloud.databricks.com
  # Enter token: dapi_xxxxxxxxxx

  databricks fs cp -r data/bronze/laws/BNS/json/ dbfs:/FileStore/nyaya/bronze/laws/BNS/json/
  databricks fs cp -r data/bronze/laws/BNSS/json/ dbfs:/FileStore/nyaya/bronze/laws/BNSS/json/
  databricks fs cp -r data/bronze/laws/BSA/json/ dbfs:/FileStore/nyaya/bronze/laws/BSA/json/
  ```

### Step 3: Import Notebooks

Import these files as **Databricks notebooks** (they use the `# Databricks notebook source` format):

| Notebook | Purpose |
|----------|---------|
| `notebooks/db_01_bronze_ingest.py` | JSON → Bronze Delta table |
| `notebooks/db_02_silver_chunk.py` | Bronze → Silver Delta (with Z-ORDER) |
| `notebooks/db_03_gold_embed.py` | Silver → Gold Delta + embeddings |

Import via: **Workspace → Import → File** → select the `.py` file.

### Step 4: Set Your OpenAI API Key

In notebook `db_03_gold_embed.py`, set the API key. Options:

1. **Notebook widget** (recommended):
   ```python
   dbutils.widgets.text("openai_api_key", "", "OpenAI API Key")
   OPENAI_API_KEY = dbutils.widgets.get("openai_api_key")
   ```

2. **Databricks Secrets** (most secure, paid tiers):
   ```python
   OPENAI_API_KEY = dbutils.secrets.get(scope="nyaya", key="openai_api_key")
   ```

3. **Environment variable** (cluster config):
   Cluster → Edit → Advanced → Spark Config:
   ```
   spark.executorEnv.OPENAI_API_KEY sk-proj-xxxxx
   ```

### Step 5: Run Notebooks in Order

1. **db_01_bronze_ingest** → creates `nyaya.bronze_laws_raw`
2. **db_02_silver_chunk** → creates `nyaya.silver_law_chunks` (Z-ORDERed)
3. **db_03_gold_embed** → creates `nyaya.gold_law_embeddings` (Z-ORDERed)

### Step 6: Verify Tables

```sql
-- In a SQL cell or SQL editor:
SELECT act_code, COUNT(*) FROM nyaya.bronze_laws_raw GROUP BY act_code;
SELECT act_code, COUNT(*) FROM nyaya.silver_law_chunks GROUP BY act_code;
SELECT act_code, COUNT(*) FROM nyaya.gold_law_embeddings GROUP BY act_code;
```

---

## What Z-ORDER Does

In notebooks `db_02` and `db_03`, we apply:

```sql
OPTIMIZE delta.`/delta/nyaya/silver/law_chunks` ZORDER BY (act_code, section_number)
```

This **physically reorders the Parquet files** so that rows with the same `act_code` and `section_number` are stored together. Benefits:

- **Data skipping**: queries like `WHERE act_code = 'BNS' AND section_number = '300'` skip 90%+ of files
- **Faster filters**: metadata lookups that filter by act + section become I/O efficient
- **Better compression**: similar data co-located → smaller files

---

## Directory Structure

```
nyaya/
├── .env                          # local environment config
├── .env.example                  # template
├── build_indexes.py              # entry point (--medallion flag)
├── requirements.txt
├── PIPELINE.md                   # ← you are here
│
├── core/
│   ├── config.py                 # all settings (medallion paths, Databricks, etc.)
│   └── data_models.py            # Pydantic models
│
├── pipelines/                    # ★ Medallion pipeline (local)
│   ├── bronze_ingest.py          # Stage 1: JSON → Bronze Delta
│   ├── silver_chunk.py           # Stage 2: Bronze → Silver Delta
│   ├── gold_embed.py             # Stage 3: Silver → Gold Delta + vector DB
│   └── run_all.py                # Orchestrator (run all or single stage)
│
├── notebooks/                    # ★ Databricks notebooks (PySpark + Delta)
│   ├── db_01_bronze_ingest.py    # Stage 1 (Spark)
│   ├── db_02_silver_chunk.py     # Stage 2 (Spark + Z-ORDER)
│   └── db_03_gold_embed.py       # Stage 3 (Spark + OpenAI + Z-ORDER)
│
├── scripts/
│   └── scrape_acts.py            # Download act data from advocatekhoj.com
│
├── rag/                          # Retrieval pipeline
│   ├── retriever.py              # Smart search (section extraction + semantic)
│   ├── vector_db.py              # ChromaDB/FAISS abstraction
│   ├── chunker.py                # Text chunking
│   ├── pipeline.py               # RAG: retrieve → prompt → LLM
│   └── prompts.py                # Prompt templates
│
├── models/
│   ├── embeddings/embedder.py    # OpenAI + HuggingFace embeddings
│   └── llm/router.py             # LLM provider routing
│
├── agents/
│   ├── orchestrator.py           # Query routing
│   └── legal_agent.py            # Legal RAG agent
│
├── app/
│   └── main.py                   # Streamlit UI
│
└── data/
    ├── bronze/
    │   ├── laws/{BNS,BNSS,BSA}/json/   # raw scraped files
    │   └── delta/laws_raw/              # ★ Bronze Delta table
    ├── silver/
    │   └── delta/law_chunks/            # ★ Silver Delta table
    └── gold/
        ├── delta/law_embeddings/        # ★ Gold Delta table
        ├── chromadb/                    # ChromaDB vector index
        └── faiss/                       # FAISS indexes (optional)
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NYAYA_ENV` | `local` | `local` or `databricks` |
| `BRONZE_DELTA_PATH` | `data/bronze/delta/laws_raw` | Bronze Delta table path |
| `SILVER_DELTA_PATH` | `data/silver/delta/law_chunks` | Silver Delta table path |
| `GOLD_DELTA_PATH` | `data/gold/delta/law_embeddings` | Gold Delta table path |
| `VECTOR_DB` | `chroma_local` | `chroma_local`, `chroma_remote`, or `faiss` |
| `OPENAI_API_KEY` | — | **Required** for embeddings + LLM |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (1536-dim) |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `huggingface` |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of results to retrieve |
| `DATABRICKS_HOST` | — | Databricks workspace URL |
| `DATABRICKS_TOKEN` | — | Databricks personal access token |
| `MLFLOW_TRACKING_URI` | — | `databricks` or local MLflow URI |
