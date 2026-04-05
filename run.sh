#!/usr/bin/env bash
# =============================================================================
# Nyaya-Sahayak — Local Run Script
# =============================================================================
# Prerequisites:
#   1. conda environment "nb" must exist (python 3.12)
#      conda create -n nb python=3.12 -y
#   2. .env file must be configured with your OPENAI_API_KEY
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                     # Run ALL steps
#   ./run.sh --skip-install      # Skip pip install (already done)
#   ./run.sh --skip-pipeline     # Skip medallion (use existing ChromaDB data)
#   ./run.sh --app-only          # Just launch Streamlit (requires indexed data)
# =============================================================================

set -e  # Exit on any error

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log_step()  { echo -e "\n${BOLD}${BLUE}══ STEP $1 ═══════════════════════════════════════${RESET}"; }
log_info()  { echo -e "${CYAN}▶ $1${RESET}"; }
log_ok()    { echo -e "${GREEN}✅ $1${RESET}"; }
log_warn()  { echo -e "${YELLOW}⚠  $1${RESET}"; }
log_error() { echo -e "${RED}❌ $1${RESET}"; }

# ── Flags ──────────────────────────────────────────────────────────────────
SKIP_INSTALL=false
SKIP_PIPELINE=false
APP_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-install)   SKIP_INSTALL=true ;;
        --skip-pipeline)  SKIP_PIPELINE=true ;;
        --app-only)       APP_ONLY=true; SKIP_INSTALL=true; SKIP_PIPELINE=true ;;
    esac
done

# ── Project root (directory where this script lives) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "\n${BOLD}${GREEN}⚖️  Nyaya-Sahayak — Local Setup & Run${RESET}"
echo -e "${CYAN}Project root: ${SCRIPT_DIR}${RESET}"
echo -e "${CYAN}$(date)${RESET}\n"

# =============================================================================
# STEP 0 — Verify conda environment exists
# =============================================================================
log_step "0: Verify conda environment 'nb'"

if ! conda info --envs | grep -q "^nb "; then
    log_error "Conda environment 'nb' not found!"
    echo "Create it with:"
    echo "  conda create -n nb python=3.12 -y"
    exit 1
fi
log_ok "Conda environment 'nb' found"

# Ensure .env exists
if [ ! -f ".env" ]; then
    log_error ".env file not found! Copy .env.example and fill in your OPENAI_API_KEY."
    exit 1
fi

# Check OPENAI_API_KEY is set in .env
if grep -q "^OPENAI_API_KEY=sk-" .env; then
    log_ok ".env found with OpenAI API key"
else
    log_warn ".env found but OPENAI_API_KEY may not be set. Continuing..."
fi

# =============================================================================
# STEP 1 — Install Python dependencies
# =============================================================================
if [ "$SKIP_INSTALL" = false ]; then
    log_step "1: Install Python dependencies"
    log_info "Running: conda run -n nb pip install -r requirements.txt"
    conda run -n nb pip install -r requirements.txt
    log_ok "Python dependencies installed"
else
    log_warn "Skipping Step 1 (--skip-install)"
fi

# =============================================================================
# STEP 2 — Verify imports (quick sanity check)
# =============================================================================
log_step "2: Verify core imports"

conda run -n nb python -c "
import sys
sys.path.insert(0, '.')

errors = []

try:
    import openai
    print('  ✅ openai')
except ImportError as e:
    errors.append(f'openai: {e}')

try:
    import chromadb
    print('  ✅ chromadb')
except ImportError as e:
    errors.append(f'chromadb: {e}')

try:
    import deltalake
    print('  ✅ deltalake')
except ImportError as e:
    errors.append(f'deltalake: {e}')

try:
    import streamlit
    print('  ✅ streamlit')
except ImportError as e:
    errors.append(f'streamlit: {e}')

try:
    from core.config import NYAYA_ENV, VECTOR_DB, OPENAI_API_KEY, OPENAI_MODEL
    print(f'  ✅ core.config (env={NYAYA_ENV}, vector_db={VECTOR_DB}, model={OPENAI_MODEL})')
    if not OPENAI_API_KEY:
        errors.append('OPENAI_API_KEY is empty in .env!')
except Exception as e:
    errors.append(f'core.config: {e}')

if errors:
    print('\\n❌ Import errors:')
    for err in errors:
        print(f'   - {err}')
    sys.exit(1)
print('\\n✅ All core imports OK')
"
log_ok "Import verification passed"

# =============================================================================
# STEP 3 — Test OpenAI API connection
# =============================================================================
log_step "3: Test OpenAI API connection"

conda run -n nb python -c "
import sys
sys.path.insert(0, '.')
from core.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Test embedding
resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=['test'])
print(f'  ✅ Embedding OK — dim={len(resp.data[0].embedding)}')

# Test chat
resp2 = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{'role':'user','content':'Say OK'}],
    max_tokens=5,
)
print(f'  ✅ Chat OK — response: {resp2.choices[0].message.content.strip()}')
print(f'  Model: {OPENAI_MODEL}, Embedding: {OPENAI_EMBEDDING_MODEL}')
"
log_ok "OpenAI API connection verified"

# =============================================================================
# STEP 4 — Create required directories
# =============================================================================
log_step "4: Create data directories"

mkdir -p data/bronze/delta/laws_raw
mkdir -p data/silver/delta/law_chunks
mkdir -p data/gold/delta/law_embeddings
mkdir -p data/gold/chromadb

log_ok "Data directories ready"

# =============================================================================
# STEP 5 — Bronze: Ingest raw JSON → Delta table
# =============================================================================
if [ "$SKIP_PIPELINE" = false ] && [ "$APP_ONLY" = false ]; then
    log_step "5: Bronze — Ingest raw JSON → Delta"

    # Check if raw data exists
    BNS_COUNT=$(ls data/bronze/laws/BNS/json/*.json 2>/dev/null | wc -l)
    BNSS_COUNT=$(ls data/bronze/laws/BNSS/json/*.json 2>/dev/null | wc -l)
    BSA_COUNT=$(ls data/bronze/laws/BSA/json/*.json 2>/dev/null | wc -l)

    if [ "$BNS_COUNT" -eq 0 ] && [ "$BNSS_COUNT" -eq 0 ] && [ "$BSA_COUNT" -eq 0 ]; then
        log_warn "No raw JSON data found in data/bronze/laws/{BNS,BNSS,BSA}/json/"
        log_warn "Skipping Bronze step. Run scripts/scrape_acts.py first to download data."
        log_warn "  conda run -n nb python scripts/scrape_acts.py"
    else
        log_info "Found: BNS=$BNS_COUNT, BNSS=$BNSS_COUNT, BSA=$BSA_COUNT JSON files"
        conda run -n nb python -m pipelines.bronze_ingest
        log_ok "Bronze Delta table created"
    fi
else
    log_warn "Skipping Step 5 (Bronze pipeline)"
fi

# =============================================================================
# STEP 6 — Silver: Chunk sections → Delta table
# =============================================================================
if [ "$SKIP_PIPELINE" = false ] && [ "$APP_ONLY" = false ]; then
    log_step "6: Silver — Chunk sections → Delta"

    BRONZE_EXISTS=$(conda run -n nb python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from core.config import BRONZE_DELTA_PATH
p = Path(BRONZE_DELTA_PATH)
print('yes' if (p / '_delta_log').exists() else 'no')
" 2>/dev/null || echo "no")

    if [ "$BRONZE_EXISTS" = "yes" ]; then
        conda run -n nb python -m pipelines.silver_chunk
        log_ok "Silver Delta table created"
    else
        log_warn "Bronze Delta not found — skipping Silver step"
    fi
else
    log_warn "Skipping Step 6 (Silver pipeline)"
fi

# =============================================================================
# STEP 7 — Gold: Embed chunks → ChromaDB (OpenAI embeddings)
# =============================================================================
if [ "$SKIP_PIPELINE" = false ] && [ "$APP_ONLY" = false ]; then
    log_step "7: Gold — Embed chunks → ChromaDB (OpenAI API)"

    SILVER_EXISTS=$(conda run -n nb python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from core.config import SILVER_DELTA_PATH
p = Path(SILVER_DELTA_PATH)
print('yes' if (p / '_delta_log').exists() else 'no')
" 2>/dev/null || echo "no")

    if [ "$SILVER_EXISTS" = "yes" ]; then
        log_info "This step calls OpenAI API to embed chunks. Cost: ~1000 chunks × \$0.00002/chunk = ~\$0.02"
        conda run -n nb python -m pipelines.gold_embed
        log_ok "Gold Delta table + ChromaDB vectors created"
    else
        log_warn "Silver Delta not found — skipping Gold/embed step"
    fi
else
    log_warn "Skipping Step 7 (Gold embed pipeline)"
fi

# =============================================================================
# STEP 8 — Verify ChromaDB has vectors
# =============================================================================
log_step "8: Verify ChromaDB vector store"

conda run -n nb python -c "
import sys
sys.path.insert(0, '.')
from core.config import CHROMA_LOCAL_DIR, VECTOR_DB
import chromadb

print(f'  Vector DB: {VECTOR_DB}')
print(f'  ChromaDB path: {CHROMA_LOCAL_DIR}')

client = chromadb.PersistentClient(path=CHROMA_LOCAL_DIR)
collections = client.list_collections()
total = 0
for c in collections:
    col = client.get_collection(c.name)
    cnt = col.count()
    total += cnt
    print(f'  Collection: {c.name} — {cnt} vectors')

if total == 0:
    print('  ⚠️  No vectors found! Run the medallion pipeline first.')
    print('     (Try: ./run.sh --skip-install)')
else:
    print(f'  ✅ Total vectors: {total}')
"

# =============================================================================
# STEP 9 — Test end-to-end RAG (quick smoke test)
# =============================================================================
log_step "9: Smoke test — end-to-end RAG query"

conda run -n nb python -c "
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.WARNING)

from core.data_models import UserQuery, Persona, ActFilter
from agents.orchestrator import handle

query = UserQuery(
    text='What is BNS Section 103?',
    user_lang='en',
    persona=Persona.CITIZEN,
    act_filter=ActFilter.ALL,
)
print(f'  Query: {query.text}')
response = handle(query)
answer_preview = response.answer_text[:300].replace('\n', ' ')
print(f'  Answer (preview): {answer_preview}...')
print(f'  Citations: {len(response.citations)}')
print(f'  Model: {response.model_ids_used}')
print('  ✅ RAG pipeline working')
" && log_ok "Smoke test passed" || log_warn "Smoke test failed (check logs, but UI may still work)"

# =============================================================================
# STEP 10 — Launch Streamlit UI
# =============================================================================
log_step "10: Launch Streamlit UI"

echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   ⚖️  Nyaya-Sahayak UI starting...                 ║"
echo "  ║   Open in browser: http://localhost:8501          ║"
echo "  ║   Press Ctrl+C to stop                           ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

conda run -n nb python -m streamlit run app/main.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false
