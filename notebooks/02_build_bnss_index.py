# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Build BNSS (Bharatiya Nagarik Suraksha Sanhita) FAISS Index
# MAGIC
# MAGIC 1. Loads all BNSS section files from `data/bronze/laws/BNSS/`
# MAGIC 2. Chunks them (~800 words, 150-word overlap)
# MAGIC 3. Embeds with `sentence-transformers/all-MiniLM-L6-v2` (CPU)
# MAGIC 4. Builds a FAISS inner-product index
# MAGIC 5. Saves to `data/gold/faiss/bnss/`

# COMMAND ----------

import sys
from pathlib import Path

ROOT = str(Path.cwd().parent) if Path.cwd().name == 'notebooks' else str(Path.cwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f'Project root: {ROOT}')

# COMMAND ----------

from core.config import ACTS
from rag.chunker import chunk_act, load_act_sections

ACT = 'BNSS'
info = ACTS[ACT]
print(f'Act: {info["name"]}')
print(f'Data dir: {info["data_dir"]}')
print(f'FAISS dir: {info["faiss_dir"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load raw sections

# COMMAND ----------

sections = load_act_sections(ACT)
print(f'Loaded {len(sections)} sections')
if sections:
    s = sections[0]
    print(f'\nSection {s["section_number"]}: {s["title"]}')
    print(s['content'][:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunk sections

# COMMAND ----------

chunks = chunk_act(ACT)
print(f'Total chunks: {len(chunks)}')
lengths = [len(c['text'].split()) for c in chunks]
print(f'Word counts: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}')
if chunks:
    print(f'\nFirst chunk: {chunks[0]["chunk_id"]}')
    print(chunks[0]['text'][:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build FAISS index

# COMMAND ----------

from rag.retriever import build_index, save_index

index, metadata = build_index(chunks, ACT)
print(f'Index vectors: {index.ntotal}')
print(f'Metadata entries: {len(metadata)}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save to disk

# COMMAND ----------

save_index(ACT)
print(f'Saved to {info["faiss_dir"]}')
for f in info['faiss_dir'].iterdir():
    print(f'  {f.name} ({f.stat().st_size / 1024:.1f} KB)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test search

# COMMAND ----------

from rag.retriever import search

for q in ['How to file an FIR?', 'Procedure for arrest', 'Provisions for anticipatory bail']:
    print(f'\n--- Query: {q} ---')
    for r in search(ACT, q, top_k=3):
        print(f'  §{r["section_number"]} {r["title"]} (score={r["score"]:.3f})')
        print(f'  {r["text"][:150]}...')