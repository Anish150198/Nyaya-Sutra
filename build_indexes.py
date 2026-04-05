"""
Build all vector DB indexes.

Two modes:
  python build_indexes.py                  # quick: JSON → chunk → embed → ChromaDB directly
  python build_indexes.py --medallion      # medallion: Bronze → Silver → Gold Delta + ChromaDB

Respects VECTOR_DB in .env:
  chroma_local  → persistent ChromaDB on disk (default for local)
  chroma_remote → ChromaDB client→server
  databricks_vs → Databricks Vector Search (use on Databricks only)
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from core.config import ACTS, VECTOR_DB
from rag.vector_db import get_db_info


def run_quick():
    """Quick mode: JSON files → chunk → embed → vector DB directly (no Delta)."""
    from rag.chunker import chunk_act
    from models.embeddings.embedder import embed_texts
    from rag.vector_db import upsert_act
    import numpy as np

    print(f"Vector DB backend: {VECTOR_DB}")

    for act_code, info in ACTS.items():
        data_dir = info["data_dir"]
        if data_dir is None or not Path(data_dir).exists():
            print(f"⚠  Skipping {act_code}: data dir not found at {data_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Building index for {act_code} ({info['name']})")
        print(f"  Backend: {VECTOR_DB}")
        print(f"{'='*60}")

        chunks = chunk_act(act_code)
        if not chunks:
            print(f"⚠  No chunks produced for {act_code}, skipping")
            continue

        texts = [c["text"] for c in chunks]
        print(f"  Embedding {len(texts)} chunks via OpenAI...")
        embeddings = embed_texts(texts, show_progress=True)

        upsert_act(act_code, chunks, embeddings)
        print(f"✓  {act_code}: {len(chunks)} chunks indexed into {VECTOR_DB}")

    print(f"\n✅ All indexes built ({VECTOR_DB}).")
    print(f"DB info: {get_db_info()}")


def run_medallion():
    """Medallion mode: Bronze → Silver → Gold Delta + vector DB."""
    from pipelines.run_all import run_bronze, run_silver, run_gold
    run_bronze()
    run_silver()
    run_gold()
    print(f"\n✅ Medallion pipeline complete ({VECTOR_DB}).")
    print(f"DB info: {get_db_info()}")


def main():
    parser = argparse.ArgumentParser(description="Build Nyaya-Sahayak indexes")
    parser.add_argument("--medallion", action="store_true",
                        help="Use medallion pipeline (Bronze → Silver → Gold Delta)")
    args = parser.parse_args()

    if args.medallion:
        run_medallion()
    else:
        run_quick()


if __name__ == "__main__":
    main()

