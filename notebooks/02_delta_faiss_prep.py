"""
02_delta_faiss_prep.py
======================
Cleans Bronze data into Silver tables, chunks legal text,
creates Gold Delta tables, and builds FAISS indexes.

Silver tables created:
  - law_sections, schemes_curated, ipc_bns_map, legal_aid_rules, bb_legal_questions
Gold tables/artifacts:
  - law_chunks, law_chunks_embedded
  - FAISS indexes: legal.index, scheme.index
"""

# COMMAND ----------
# %pip install faiss-cpu sentence-transformers

# COMMAND ----------

import sys
import re
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path("__file__").resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from core.config import (
    SILVER_SCHEMA, GOLD_SCHEMA,
    TABLE_LAWS_RAW, TABLE_SCHEMES_RAW, TABLE_IPC_BNS_RAW,
    TABLE_LAW_SECTIONS, TABLE_SCHEMES_CURATED, TABLE_IPC_BNS_MAP,
    TABLE_LAW_CHUNKS, TABLE_LAW_CHUNKS_EMBEDDED,
    FAISS_LEGAL_INDEX_PATH, FAISS_SCHEME_INDEX_PATH,
)

# COMMAND ----------
try:
    spark  # noqa
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("NyayaSutra_DeltaFAISS").getOrCreate()

# COMMAND ----------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")

# COMMAND ----------
# ========================
# SILVER: Clean & Structure
# ========================

# --- Law Sections ---
print("Processing law_sections...")
try:
    laws_raw = spark.table(TABLE_LAWS_RAW).toPandas()

    # Expected columns after parsing: code, chapter, section_no, title, text, language, act_type
    # Adjust parsing logic based on your actual raw data format
    if "text" in laws_raw.columns and "section_no" in laws_raw.columns:
        law_sections_df = spark.createDataFrame(laws_raw)
        law_sections_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_LAW_SECTIONS)
        print(f"  ✓ {TABLE_LAW_SECTIONS}: {law_sections_df.count()} rows")
    else:
        print(f"  ⚠ Raw law data missing expected columns. Available: {list(laws_raw.columns)}")
except Exception as e:
    print(f"  ⚠ Could not process laws: {e}")

# --- Schemes Curated ---
print("Processing schemes_curated...")
try:
    schemes_raw = spark.table(TABLE_SCHEMES_RAW).toPandas()
    # Clean and add typed columns for filtering
    # Adjust based on your actual raw scheme data format
    schemes_df = spark.createDataFrame(schemes_raw)
    schemes_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_SCHEMES_CURATED)
    print(f"  ✓ {TABLE_SCHEMES_CURATED}: {schemes_df.count()} rows")
except Exception as e:
    print(f"  ⚠ Could not process schemes: {e}")

# --- IPC→BNS Mapping ---
print("Processing ipc_bns_map...")
try:
    mapping_raw = spark.table(TABLE_IPC_BNS_RAW).toPandas()
    mapping_df = spark.createDataFrame(mapping_raw)
    mapping_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_IPC_BNS_MAP)
    print(f"  ✓ {TABLE_IPC_BNS_MAP}: {mapping_df.count()} rows")
except Exception as e:
    print(f"  ⚠ Could not process IPC→BNS mapping: {e}")

# COMMAND ----------
# ========================
# GOLD: Chunking & Embeddings
# ========================

print("\nChunking law sections...")

CHUNK_SIZE = 512   # tokens (approximate with words for now)
CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


try:
    law_sections = spark.table(TABLE_LAW_SECTIONS).toPandas()
    chunk_records = []
    for _, row in law_sections.iterrows():
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "chunk_id": f"{row.get('id', '')}_{i}",
                "section_id": row.get("id", ""),
                "code": row.get("code", ""),
                "section_no": row.get("section_no", ""),
                "title": row.get("title", ""),
                "text_chunk": chunk,
                "lang": row.get("language", "en"),
                "act_type": row.get("act_type", ""),
            })

    if chunk_records:
        import pandas as pd
        chunks_pdf = pd.DataFrame(chunk_records)
        chunks_df = spark.createDataFrame(chunks_pdf)
        chunks_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_LAW_CHUNKS)
        print(f"  ✓ {TABLE_LAW_CHUNKS}: {len(chunk_records)} chunks from {len(law_sections)} sections")
    else:
        print("  ⚠ No chunks generated (check law_sections data)")
except Exception as e:
    print(f"  ⚠ Chunking failed: {e}")

# COMMAND ----------
# ========================
# GOLD: Build FAISS Indexes
# ========================

print("\nBuilding FAISS indexes...")

try:
    import faiss
    from models.embeddings.embedder import embed_texts, get_embedding_dim

    # --- Legal Index ---
    chunks_pdf = spark.table(TABLE_LAW_CHUNKS).toPandas()
    texts = chunks_pdf["text_chunk"].tolist()

    if texts:
        print(f"  Embedding {len(texts)} legal chunks...")
        embeddings = embed_texts(texts, show_progress=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim on normalized vecs)
        index.add(embeddings)

        # Save index
        idx_path = Path(FAISS_LEGAL_INDEX_PATH)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(idx_path))

        # Save metadata
        metadata = chunks_pdf[["chunk_id", "code", "section_no", "title", "text_chunk"]].rename(
            columns={"text_chunk": "text"}
        ).to_dict("records")
        with open(idx_path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"  ✓ Legal FAISS index saved: {index.ntotal} vectors → {idx_path}")
    else:
        print("  ⚠ No legal chunks to index")

    # --- Scheme Index ---
    try:
        schemes_pdf = spark.table(TABLE_SCHEMES_CURATED).toPandas()
        scheme_texts = (
            schemes_pdf.get("name", "").astype(str) + " " +
            schemes_pdf.get("description", "").astype(str) + " " +
            schemes_pdf.get("eligibility_text", "").astype(str)
        ).tolist()

        if scheme_texts:
            print(f"  Embedding {len(scheme_texts)} schemes...")
            scheme_embeddings = embed_texts(scheme_texts, show_progress=True)

            scheme_index = faiss.IndexFlatIP(scheme_embeddings.shape[1])
            scheme_index.add(scheme_embeddings)

            scheme_idx_path = Path(FAISS_SCHEME_INDEX_PATH)
            scheme_idx_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(scheme_index, str(scheme_idx_path))

            scheme_meta = schemes_pdf.to_dict("records")
            with open(scheme_idx_path.with_suffix(".meta.pkl"), "wb") as f:
                pickle.dump(scheme_meta, f)

            print(f"  ✓ Scheme FAISS index saved: {scheme_index.ntotal} vectors → {scheme_idx_path}")
        else:
            print("  ⚠ No schemes to index")
    except Exception as e:
        print(f"  ⚠ Scheme indexing failed: {e}")

except ImportError:
    print("  ⚠ faiss-cpu or sentence-transformers not installed. Run: pip install faiss-cpu sentence-transformers")
except Exception as e:
    print(f"  ⚠ FAISS index build failed: {e}")

# COMMAND ----------
print("\n=== Delta/FAISS Preparation Complete ===")
