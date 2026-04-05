# Nyaya-Sahayak — Setup & Run Guide

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│   Orchestrator      │  ← agents/orchestrator.py
│   Agent             │
└─────┬───────────────┘
      │ classify intent → LEGAL / WELFARE / MIXED / GENERIC
      ▼
┌─────────────────────┐
│   Legal Agent       │  ← agents/legal_agent.py
└─────┬───────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│   RAG Pipeline (rag/pipeline.py)                        │
│                                                         │
│   ┌─────────────────────────────────────────────┐       │
│   │ Vector DB  (rag/vector_db.py)               │       │
│   │                                             │       │
│   │  chroma_local  → persistent on-disk DB      │       │
│   │  chroma_remote → HTTP client→server         │       │
│   │  faiss         → flat files (lightweight)   │       │
│   │                                             │       │
│   │  Collection: nyaya_bns   ──┐                │       │
│   │  Collection: nyaya_bnss  ──┼→ top-k chunks  │       │
│   │  Collection: nyaya_bsa   ──┘                │       │
│   └─────────────────────────────────────────────┘       │
│                              │                          │
│                              ▼                          │
│                     prompt template                     │
│                   + LLM (OpenAI / local / none)         │
│                              │                          │
│                              ▼                          │
│                         Answer + Sources                │
└─────────────────────────────────────────────────────────┘
```

Each act has its **own** vector collection/index. The classifier auto-detects
which act(s) to search, or the user can force a filter from the sidebar.

---

## 1. Quick Start (5 minutes)

### Prerequisites

- Python 3.10+ (tested on 3.12)
- ~2 GB disk for embedding model cache + indexes
- OpenAI API key (for LLM answers)

### Install

```bash
cd /home/akash/Downloads/nyaya/test/test

# Create venv (or use conda)
python -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-xxxxxxxxxxxx
#   VECTOR_DB=chroma_local          (default, recommended)
```

### Build vector indexes (one-time, ~2 min)

```bash
python build_indexes.py
```

This will:
1. Load all JSON/TXT section files per act (BNS=360, BNSS=539, BSA=172)
2. Chunk them (~800 words, 150-word overlap)
3. Embed with `sentence-transformers/all-MiniLM-L6-v2` (~90MB download on first run)
4. Store into the configured vector DB (ChromaDB or FAISS)

**Or build individually via notebooks:**
```
notebooks/01_build_bns_index.ipynb
notebooks/02_build_bnss_index.ipynb
notebooks/03_build_bsa_index.ipynb
```

### Run the Streamlit app

```bash
streamlit run app/main.py
```

Open http://localhost:8501 and start asking questions.

---

## 2. Vector DB Configuration

The vector DB backend is set via `VECTOR_DB` in `.env`. Three options:

### Option A: ChromaDB Local (default, recommended)

Persistent on-disk database. No server needed. Data survives restarts.

```env
VECTOR_DB=chroma_local
CHROMA_LOCAL_DIR=./data/gold/chromadb
```

Data is stored at `data/gold/chromadb/`. Each act gets a separate collection:
- `nyaya_bns` — BNS sections
- `nyaya_bnss` — BNSS sections
- `nyaya_bsa` — BSA sections

### Option B: ChromaDB Remote (for team/production use)

Connect to a shared ChromaDB server running on another host.

**Start a ChromaDB server** (on the remote machine):
```bash
pip install chromadb
chroma run --host 0.0.0.0 --port 8000 --path /data/chromadb
```

**Or via Docker:**
```bash
docker run -d --name chromadb \
  -p 8000:8000 \
  -v /data/chromadb:/chroma/chroma \
  chromadb/chroma:latest
```

**Client config** in `.env`:
```env
VECTOR_DB=chroma_remote
CHROMA_HOST=your-server-ip       # or localhost
CHROMA_PORT=8000
CHROMA_AUTH_TOKEN=               # optional Bearer token
CHROMA_SSL=false                 # set true if behind HTTPS
```

Then build indexes as usual — they'll be stored on the remote server:
```bash
python build_indexes.py
```

### Option C: FAISS flat files (lightweight, no server)

Minimal: just flat `.faiss` + `.json` files on disk. Good for quick experiments.

```env
VECTOR_DB=faiss
```

Data saved to `data/gold/faiss/{bns,bnss,bsa}/`.

### Comparison

| Feature | chroma_local | chroma_remote | faiss |
|---------|-------------|---------------|-------|
| Persistence | Disk (auto) | Server (auto) | Flat files |
| Server needed | No | Yes | No |
| Shared access | No | Yes (multi-client) | No |
| Metadata filtering | Built-in | Built-in | Manual |
| Production ready | Dev/single-user | Yes | No |
| Dependencies | `chromadb` | `chromadb` | `faiss-cpu` |

---

## 3. LLM Configuration

### Option A: OpenAI (recommended to start)

In `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini          # or gpt-4o, gpt-3.5-turbo
```

### Option B: Local GGUF model (CPU-only, for later)

```bash
pip install llama-cpp-python

mkdir -p models/weights
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf --local-dir models/weights
```

In `.env`:
```env
LLM_PROVIDER=local
LOCAL_LLM_PATH=./models/weights/llama-2-7b-chat.Q4_K_M.gguf
```

### Option C: No LLM (context-only mode)

If neither `OPENAI_API_KEY` nor `LOCAL_LLM_PATH` is set, the pipeline
returns the raw retrieved legal sections without LLM summarization.
Still useful for verifying search quality.

---

## 4. Project Structure

```
test/test/
├── build_indexes.py             # One-command: build all vector DB indexes
├── .env.example                 # Environment variable template
├── requirements.txt
│
├── core/
│   ├── config.py                # Paths, acts registry, vector DB + LLM settings
│   └── data_models.py           # Pydantic models
│
├── rag/
│   ├── vector_db.py             # Vector DB abstraction (Chroma local/remote, FAISS)
│   ├── chunker.py               # Load JSON/TXT → chunk per act
│   ├── retriever.py             # Search API (delegates to vector_db.py)
│   ├── prompts.py               # Citizen & lawyer prompt templates
│   └── pipeline.py              # retrieve → prompt → generate
│
├── models/
│   ├── embeddings/embedder.py   # sentence-transformers (CPU)
│   ├── llm/router.py            # OpenAI / local / none selector
│   ├── llm/openai_runner.py     # OpenAI API wrapper
│   ├── llm/local_runner.py      # llama-cpp-python CPU wrapper
│   └── nlp_classifier/          # Intent classifier (keyword-based)
│
├── agents/
│   ├── orchestrator.py          # Main entry point, routes to agents
│   ├── legal_agent.py           # Legal query handler
│   └── ...                      # guardrail, welfare, translation agents
│
├── app/
│   └── main.py                  # Streamlit UI
│
├── notebooks/
│   ├── 01_build_bns_index.ipynb
│   ├── 02_build_bnss_index.ipynb
│   ├── 03_build_bsa_index.ipynb
│   └── 04_query_test.ipynb
│
└── data/
    ├── bronze/laws/
    │   ├── BNS/json/            # 360 section JSONs
    │   ├── BNSS/{json,txt}/     # 539 sections
    │   └── BSA/                 # 172 sections (json + txt)
    └── gold/
        ├── chromadb/            # ChromaDB persistent storage (default)
        └── faiss/               # FAISS flat files (alternative)
            ├── bns/
            ├── bnss/
            └── bsa/
```

---

## 5. How the Pipeline Works

1. **User asks**: "What is the punishment for theft?"

2. **Orchestrator** checks which vector collections exist, routes to legal agent

3. **Legal Agent** calls RAG pipeline with act filter

4. **RAG Pipeline**:
   - Queries the vector DB (ChromaDB or FAISS) for top-k similar chunks
   - Formats retrieved sections into context
   - Builds persona-aware prompt (citizen = simple, lawyer = formal)
   - Sends to LLM (OpenAI API / local GGUF / or returns raw context)

5. **Response** includes: answer text, source sections with scores, disclaimer

---

## 6. Databricks Deployment (Optional)

### Upload data to Volumes

```python
# In a Databricks notebook:
dbutils.fs.cp("file:///path/to/data/bronze/laws/",
              "dbfs:/Volumes/nyaya_sahayak/data/bronze/laws/", recurse=True)
```

### Build indexes on Databricks

```python
%pip install chromadb faiss-cpu sentence-transformers openai
import sys
sys.path.insert(0, "/Workspace/Repos/<user>/test")
from build_indexes import main
main()
```

### Deploy as Databricks App

The `app.yaml` in the project root configures the Streamlit entrypoint.

---

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env   # edit: OPENAI_API_KEY, VECTOR_DB

# Build indexes (one-time)
python build_indexes.py

# Run app
streamlit run app/main.py

# Test in notebook
jupyter notebook notebooks/04_query_test.ipynb

# Start ChromaDB server (for remote mode)
chroma run --host 0.0.0.0 --port 8000 --path ./data/gold/chromadb
```
